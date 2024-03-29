(ns scratch
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
            [tablecloth.api :as tc]))


(defn now []
  (java.util.Date.))

;; copied from the original llamma.clj tutorials

(defonce llama7b-path "/workspace/models/llama-2-7b-chat.ggmlv3.q4_0.bin")
(defonce llama-context (llama/create-context llama7b-path {}))
(def llama-eos (llama/eos llama-context))

(def token->str
  (into (sorted-map)
        (comp (map
               (fn [token]
                 [token (raw/llama_token_to_str llama-context token)]))
              (take-while (fn [[token untoken]]
                            untoken)))
        (range 0 Integer/MAX_VALUE)))

(defonce previous* (atom nil))
(defn get-logits [ctx s]
  #_(raw/llama_set_rng_seed ctx 1234)
  (cond

    (string? s)
    (llama/llama-update ctx s 0)

    (vector? s)
    (let [prev @previous*]
      (if (and
           (vector? prev)
           (= prev (butlast s)))
        (llama/llama-update ctx (last s))
        (do
          (llama/llama-update ctx (llama/bos) 0)
          (run! #(llama/llama-update ctx %)
                s)))))
  (reset! previous* s)

  (into [] (llama/get-logits ctx)))

(delay
  (->> ["Good morning."]
       (get-logits llama-context)
       argops/argmax
       token->str))

(delay
  (->> ["Good morning."
        "What is good?"]
       (get-logits llama-context)
       argops/argmax
       token->str))



(defn M-step [samplef previous-tokens]
  (prn [:M (now) (count previous-tokens)])
  (->> previous-tokens
       (get-logits llama-context)
       samplef
       (conj previous-tokens)))

(defn finished? [tokens]
  (-> tokens
      last
      (= llama-eos)))

(defn gen-samplef [seed]
  (raw/llama_set_rng_seed llama-context seed)
  (llama/init-mirostat-v2-sampler llama-context))

(delay
  (let [samplef (gen-samplef 1234)]
    (repeatedly 3
                #(->> "What is"
                      (llutil/tokenize llama-context)
                      (iterate (partial M-step samplef))
                      (take 5)
                      last
                      (llutil/untokenize llama-context)))))

(delay
  (let [samplef (gen-samplef 12345)]
    (->> #(->> "Write a short and wise poem."
               (llutil/tokenize llama-context)
               (iterate (partial M-step samplef))
               (take 500)
               (filter finished?)
               first
               (llutil/untokenize llama-context))
         (repeatedly 5)
         vec)))

(defn G [threshold current-tokens]
  (if (-> current-tokens
          (->> (llutil/untokenize llama-context))
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


#_(find-c [0.1 0.2] 10)

(defn spy [x tag]
  (prn [tag x])
  x)

(delay
  (let [samplef (gen-samplef 12345)]
    (->> "Please complete the sentence in ten words. Clojure is a"
         (llutil/tokenize llama-context)
         (M-step samplef)
         (G 9))))

(def *state (atom {:stop false
                   :particles []}))

(defn now []
  (java.util.Date.))

(delay
  (let [max-token-length 5
        N 100
        K 3
        s0 (->> "The Fed says"
                (llutil/tokenize llama-context))
        samplef (gen-samplef 123456)
        initial-N 20]
    (swap! *state
           assoc :particles  (tc/dataset {:x (repeat initial-N s0)
                                          :w 1
                                          :time (repeat initial-N (now))
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
                                                                     :time (now)
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
                                  (spy :before-normalize)
                                  (tc/add-column :w #(-> % :w normalize))
                                  (spy :after-normalize)
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
  (-> @*state
      :particles
      (tc/map-columns :finished
                      [:x]
                      finished?)
      (tc/map-columns :length
                      [:x]
                      count) (tc/map-columns :x
                                             [:x]
                                             (partial llutil/untokenize
                                                      llama-context))
      #_(tc/drop-columns [:x])
      (tech.v3.dataset.print/print-range :all)
      ((juxt identity
             #(tc/write! % "/tmp/particles.csv")))))

(comment
  (-> "/tmp/particles.csv"
      (tc/dataset {:key-fn keyword})
      :finished
      frequencies))

(java.util.Date.)
;; #inst "2023-12-16T22:22:34.961-00:00"
