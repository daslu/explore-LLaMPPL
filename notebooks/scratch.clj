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
    (->> #(->> "In one brief sentence, Clojure is a"
               (llutil/tokenize llama-context)
               (iterate (partial M-step samplef))
               (take 500)
               (filter finished?)
               first
               (llutil/untokenize llama-context))
         (repeatedly 3)
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
    (->> "In one brief sentence, Clojure is a"
         (llutil/tokenize llama-context)
         (M-step samplef)
         (G 9))))


(def results
  (let [N 10
        K 3
        s0 (->> "In one brief sentence, Clojure is a"
                (llutil/tokenize llama-context))
        samplef (gen-samplef 12345)]
    (loop [particles (tc/dataset {:x [s0]
                                  :w [1]})]
      (let [finished (->> particles
                          :x
                          (map finished?))]
        (-> finished
            frequencies
            prn)
        (if (every? true? finished)
          {:particles particles
           :Z (-> particles :w fun/mean)}
          ;; else
          (let [K (->> finished
                       (map (fn [f]
                              (if f 1 K))))
                N-prime (fun/sum K)]
            (-> particles
                (tc/add-columns {:K K
                                 :finished finished})
                (tc/rows :as-maps)
                (->> (map (fn [{:keys [x w finished K]}]
                            (if finished
                              (tc/dataset {:x [x]
                                           :w [(* w N-prime (/ N))]})
                              ;; else
                              (-> (range K)
                                  (->> (map (fn [k]
                                              (-> {:x (M-step samplef x)}))))
                                  tc/dataset
                                  (tc/map-columns
                                   :w
                                   [:x]
                                   (fn [x]
                                     (* (/ N-prime
                                           (* K N))
                                        w
                                        (G 9 x))))))))
                     (apply tc/concat))
                (spy :before-normalize)
                (tc/add-column :w #(-> % :w normalize))
                (spy :after-normalize)
                ((fn [{:keys [x w]
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
                                     (let [i (first candidates)]
                                       (if (-> i w (> U))
                                         (recur (rest candidates)
                                                (- U (w i))
                                                (conj I-strat i))
                                         (recur (rest candidates)
                                                U
                                                I-strat)))))]
                     (tc/dataset
                      (concat (->> I-det
                                   (map (fn [i]
                                          {:x (x i)
                                           :w (* (w i)
                                                 (/ N N-prime))})))
                              (->> I-strat
                                   (map (fn [i]
                                          {:x (x i)
                                           :w (* (/ N N-prime c*)
                                                 w-sum)}))))))))
                recur)))))))







;; def smc_standard(model, n_particles, ess_threshold=0.5):
;;     # Create n_particles copies of the model
;;     particles = [copy.deepcopy(model) for _ in range(n_particles)]
;;     weights = [0.0 for _ in range(n_particles)]

;;     while any(map(lambda p: not p.done_stepping(), particles)):
;;         # Step each particle
;;         for i, p in enumerate(particles):
;;             if not p.done_stepping():
;;                 p.step()
;;             print(f"Particle {i}: {p} (weight {p.weight})")

;;         # Normalize weights
;;         W = np.array([p.weight for p in particles])
;;         w_sum = logsumexp(W)
;;         normalized_weights = W - w_sum

;;         # Resample if necessary
;;         if -logsumexp(normalized_weights * 2) < np.log(ess_threshold) + np.log(n_particles):
;;             # Alternative implementation uses a multinomial distribution and only makes n-1 copies, reusing existing one, but fine for now
;;             probs = np.exp(normalized_weights)
;;             particles = [copy.deepcopy(particles[np.random.choice(range(len(particles)), p=probs)]) for _ in range(n_particles)]
;;             avg_weight = w_sum - np.log(n_particles)
;;             for p in particles:
;;                 p.weight = avg_weight

;;     # Return the particles
;;     return particles

(defn logsumexp [arr]
  (let [m (fun/reduce-max arr)]
    (-> arr
        (fun/- m)
        fun/exp
        fun/sum
        fun/log
        fun/+ m)))

#_(let [ess-threshold 0.5
        s0 (->> "In one brief sentence, Clojure is a"
                (llutil/tokenize llama-context))
        n-particles 10]
    (loop [particles (tc/dataset
                      {:x (repeat n-particles s0)
                       :w 0})]
     (let [finished (->> particles
                        :x
                        (map finished?))]
      (-> finished
          frequencies
          prn)
      (if (every? true? finished)
        {:particles particles}
        ;; else
        (let [new-particles (->> particles
                                 :x
                                 (map M-step))
              normalized-weights (-> particles
                                     :w
                                     (#(fun/- % (logsumexp %))))]
          (if (< (-> normalized-weights
                     (fun/* 2)
                     logsumexp
                     -)
                 (+ (fun/log ess-threshold)
                    (fun/log n-particles)))
            (let [probs (fun/exp normalized-weights)
                  new-particles ])))))))
