(ns llamppl.sampling
  (:require [instaparse.core :as insta]
            [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [com.phronemophobic.llama.util :as llutil]
            [com.phronemophobic.llama.impl.model :as model]
            [clojure.java.io :as io]
            [clojure.string :as string]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.functional :as fun]
            [tech.v3.datatype.argops :as argops]
            [tech.v3.dataset.print]
            [tablecloth.api :as tc]
            [llamppl.util :as util]
            [llamppl.context :as context]))


(defn M-step [samplef previous-tokens]
  #_(prn [:M (util/now) (count previous-tokens)])
  (->> previous-tokens
       context/logits
       samplef
       (conj previous-tokens)))

(defn finished? [tokens]
  (-> tokens
      last
      (= context/llama-eos)))


(delay
  (let [_ (context/init!)
        samplef (context/gen-samplef 1234)]
    (repeatedly 3
                #(->> "What is"
                      context/tokenize
                      (iterate (partial M-step samplef))
                      (take 5)
                      last
                      context/untokenize))))


(delay
  (let [_ (context/init!)
        samplef (context/gen-samplef 12345)]
    (->> #(->> "I'll just quote a poem."
               context/tokenize
               (iterate (partial M-step samplef))
               (take 50)
               (filter finished?)
               first
               context/untokenize)
         (repeatedly 1)
         vec)))

(defn G [threshold current-tokens]
  (if (-> current-tokens
          context/untokenize
          (string/split  #" ")
          (->> (every? #(-> % count (<= threshold)))))
    1 0))


(defn normalize [ws]
  (fun// ws
         (fun/sum ws)))

(defn find-c [weights N]
  (prn [:weights weights
        :N N])
  (let [sorted-weights (vec (sort weights))]
    (loop [B-val 0.0
           A-val (count weights)
           i 0]
      (let [chi (sorted-weights i)
            new-A-val (dec A-val)
            new-B-val (+ B-val chi)]
        (if (= i N)
          N
          (if (-> new-B-val
                  (/ chi)
                  (+ new-A-val)
                  (- N)
                  (<= 1e-12))
            (/ (- N new-A-val)
               new-B-val)
            (recur new-B-val
                   new-A-val
                   (inc i))))))))


(delay
  (find-c [0.1 0.2] 10))

(delay
  (let [_ (context/init!)
        samplef (context/gen-samplef 12345)]
    (->> "Please complete the sentence in ten words. Clojure is a"
         context/tokenize
         (M-step samplef)
         (M-step samplef)
         (M-step samplef)
         ((juxt context/untokenize
                (partial G 9)
                (partial G 12))))))

(def *state (atom {:stop false
                   :particles []}))

(defn run! [{:keys [cache-threshold
                    seed
                    max-token-length
                    N
                    K
                    base-text
                    initial-N]}]
  (let [_ (context/init! {:threshold cache-threshold})
        samplef (context/gen-samplef seed)
        s0 (context/tokenize base-text)]
    (swap! *state
           assoc :particles  (tc/dataset {:x (repeat initial-N s0)
                                          :w 1
                                          :time (repeat initial-N (util/now))
                                          :gen 0}))
    (loop [gen 1]
      (let [particles (:particles @*state)
            finished (->> particles
                          :x
                          (map finished?))]
        (->> finished
             frequencies
             (vector :finished-freqs)
             prn)
        (if (or (:stop @*state)
                (every? true? finished))
          {:particles particles
           :Z (-> particles :w fun/mean)}
          ;; else
          (let [K (->> finished
                       (map (fn [f]
                              (if f 1 K))))
                N-prime (fun/sum K)
                new-particles (-> particles
                                  (tc/add-columns {:K K
                                                   :finished finished})
                                  (tc/rows :as-maps)
                                  (->> (map (fn [{:keys [x w finished K]
                                                  :as row}]
                                              (if finished
                                                (tc/dataset {:x [x]
                                                             :w [(* w N-prime (/ N))]
                                                             :time [(:time row)]
                                                             :gen [(:gen row)]})
                                                ;; else
                                                (-> (range K)
                                                    (->> (map (fn [k]
                                                                (-> {:x (M-step samplef x)
                                                                     :time (util/now)
                                                                     :gen gen}))))
                                                    tc/dataset
                                                    (tc/map-columns
                                                     :w
                                                     [:x]
                                                     (fn [x]
                                                       (* (/ N-prime
                                                             (* K N))
                                                          w
                                                          (G max-token-length x))))))))
                                       (apply tc/concat))
                                  (util/spy :before-normalize)
                                  (tc/add-column :w #(-> % :w normalize))
                                  (util/spy :after-normalize)
                                  ((fn [{:keys [x w time gen]
                                         :as new-particles}]
                                     (prn [:new-particles new-particles])
                                     (let [w-sum (fun/sum w)
                                           c* (find-c w N)
                                           indexes (-> new-particles
                                                       tc/row-count
                                                       range)
                                           I-det (->> indexes
                                                      (filter (fn [i]
                                                                (-> i
                                                                    w
                                                                    (* c*)
                                                                    (>= 1)))))
                                           I-stoch (->> indexes
                                                        (filter (fn [i]
                                                                  (-> i
                                                                      w
                                                                      (* c*)
                                                                      (< 1))))
                                                        vec)
                                           alpha (/ (->> I-stoch
                                                         (map w)
                                                         fun/sum)
                                                    (- N (count I-det)))
                                           I-strat (loop [candidates I-stoch
                                                          U (* alpha (rand))
                                                          I-strat []]
                                                     (if (empty? candidates)
                                                       I-strat
                                                       (let [i (first candidates)
                                                             U (- U (w i))]
                                                         (if (neg? U)
                                                           (recur (rest candidates)
                                                                  (+ U alpha)
                                                                  (conj I-strat i))
                                                           (recur (rest candidates)
                                                                  U
                                                                  I-strat)))))]
                                       (prn [:c* c*
                                             :I-det I-det
                                             :I-stoch I-stoch
                                             :I-strat I-strat])
                                       (tc/dataset
                                        (concat (->> I-det
                                                     (map (fn [i]
                                                            {:x (x i)
                                                             :w (* (w i)
                                                                   (/ N N-prime))
                                                             :time (time i)
                                                             :gen (gen i)})))
                                                (->> I-strat
                                                     (map (fn [i]
                                                            {:x (x i)
                                                             :w (* (/ N N-prime c*)
                                                                   w-sum)
                                                             :time (time i)
                                                             :gen (gen i)})))))))))]
            (swap! *state
                   assoc :particles new-particles)
            (recur (inc gen))))))))




(delay
  (run!
   {:cache-threshold 30
    :seed 1
    :base-text "The Fed says"
    :max-token-length 5
    :N 15
    :K 3
    :initial-N 5}))




(delay
  (-> @*state
      :particles
      (tc/map-columns :finished
                      [:x]
                      finished?)
      (tc/map-columns :length
                      [:x]
                      count)
      (tc/map-columns :x
                      [:x]
                      context/untokenize)
      #_(tc/drop-columns [:x])
      (tech.v3.dataset.print/print-range :all)
      ((juxt #(tc/write! % "/tmp/particles3.csv")
             identity))))
