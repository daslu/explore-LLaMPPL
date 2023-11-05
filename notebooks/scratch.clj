(ns scratch
  (:require [instaparse.core :as insta]
            [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [com.phronemophobic.llama.util :as llutil]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [tech.v3.datatype.functional :as fun]
            [tech.v3.datatype.argops :as argops]))

;; copied from the original llamma.clj tutorials

(defonce llama7b-path "/workspace/models/llama-2-7b-chat.ggmlv3.q4_0.bin")
(defonce llama-context (llama/create-context llama7b-path {}))

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
     token->str)

(->> ["Good morning."
      "What is good?"]
     (get-logits llama-context)
     argops/argmax
     token->str)

(defn llama2-prompt
  "Meant to work with llama-2-7b-chat.ggmlv3.q4_0.bin"
  [prompt]
  (str
   "[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

" prompt " [/INST]
"))

(def response-tokens
  (loop [tokens (llutil/tokenize llama-context
                                 (llama2-prompt "describe clojure in one sentence."))]
    (let [logits (get-logits llama-context tokens)
          ;; greedy sampling
          token (->> logits
                     (map-indexed (fn [idx p]
                                    [idx p]))
                     (apply max-key second)
                     first)]
      (if (= token (llama/eos))
        tokens
        (recur (conj tokens token))))))

(llutil/untokenize llama-context response-tokens)
