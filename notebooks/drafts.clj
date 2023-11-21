



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
