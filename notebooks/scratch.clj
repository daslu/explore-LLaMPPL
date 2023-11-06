(ns scratch
  (:require [instaparse.core :as insta]
            [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [com.phronemophobic.llama.util :as llutil]
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
  (raw/llama_set_rng_seed ctx 1234)
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

(->> ["Good morning."]
     (get-logits llama-context)
     argops/argmax
     token->str
     delay)

(->> ["Good morning."
      "What is good?"]
     (get-logits llama-context)
     argops/argmax
     token->str
     delay)

(def samplef
  (llama/init-mirostat-v2-sampler llama-context))

(defn M-step [previous-tokens]
  (prn [(now) (count previous-tokens)])
  (->> previous-tokens
       (get-logits llama-context)
       samplef
       (conj previous-tokens)))

(defn finished? [tokens]
  (-> tokens
      last
      (= llama-eos)))

(->> "Very briefly, Clojure is a"
     (llutil/tokenize llama-context)
     (iterate M-step)
     (take 500)
     (filter finished?)
     first
     (llutil/untokenize llama-context))

(defn G [previous-tokens current-tokens]
  (when (-> current-tokens
            (->> (llama-context llutil/untokenize))
            (string/split  #" ")
            (->> (every? #(-> % count (<= 5)))))
    1 0))


(let [N 20
      K 3
      s0 []]
  (loop [particles (tc/dataset {:x [s0]
                                :w [1]})
         t 1]
    (let [particles1 (-> particles
                         (tc/map-columns :finished [:x] finished?))]
      (if (->> particles1 :finished (every? true?))
        {:particles particles
         :Z (fun/mean ws)}
        ;; else
        (-> particles1
            (tc/map-columns :K [:finished] :int32 #(+ (* K (- 1 %))
                                                      %))
            (tc/add-column :N-prime #(-> % :K fun/sum))
            (tc/map-columns :new-xs
                            [:x :finished :N-prime]
                            (fn [x w finished N-prime]
                              (if finished
                                [x (* w N-prime (/ N))]
                                ;; else
                                ))))
        (let [Ks
              N-prime (fun/sum Ks)
              new-xs (->> xs
                          )])))))
