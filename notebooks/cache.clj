(ns cache
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

(defmacro ttime [tag form]
  `(do (println (str "\t" ~tag))
       (print "\t  ")
       (time ~form)))

(defn test-texts [ctx text1 text2]
  (prn [:----------------])
  (do (raw/llama_set_rng_seed ctx 1234)
      (let [;; _ (ttime :bos (llama/llama-update ctx (llama/bos) 0))
            _ (ttime text1 (llama/llama-update ctx text1))
            x (-> ctx
                  llama/get-logits
                  argops/argmax
                  token->str)
            ;; _ (ttime :bos (llama/llama-update ctx (llama/bos) 0))
            _ (ttime text2 (llama/llama-update ctx text2))
            y (-> ctx
                  llama/get-logits
                  argops/argmax
                  token->str)]
        [x y])))

(let [ctx (llama/create-context llama7b-path {:f16-kv true})]
  (repeatedly
   4
   #(test-texts ctx
                "How much wood would a woodchuck chuck if a woodchuck would chuck wood?"
                "How much wood would a woodchuck chuck if a lion may chuck stone?")))
