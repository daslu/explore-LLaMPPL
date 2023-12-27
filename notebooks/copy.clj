(ns copy
  (:require [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [com.phronemophobic.llama.util :as llutil]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [tech.v3.datatype.argops :as argops]))

(def llama7b-path "/workspace/models/llama-2-7b-chat.ggmlv3.q4_0.bin")

(defn llama-copy [ctx]
  (let [mem (byte-array (raw/llama_get_state_size ctx))]
    (raw/llama_copy_state_data ctx mem)
    mem))

(defn ->token->str [ctx]
  (into (sorted-map)
        (comp (map
               (fn [token]
                 [token (raw/llama_token_to_str ctx token)]))
              (take-while (fn [[token untoken]]
                            untoken)))
        (range 0 Integer/MAX_VALUE)))

(comment
  (let [ctx (llama/create-context llama7b-path {})
        token->str (->token->str ctx)
        _ (llama/llama-update ctx (llama/bos) 0)
        _ (llama/llama-update ctx "What is the")
        mem (llama-copy ctx)]
    (-> ctx
        llama/get-logits
        argops/argmax
        token->str
        prn)
    (llama/llama-update ctx (llama/bos) 0)
    (llama/llama-update ctx "What is the matter with")
    (-> ctx
        llama/get-logits
        argops/argmax
        token->str
        prn)
    (raw/llama_set_state_data ctx mem)
    (-> ctx
        llama/get-logits
        argops/argmax
        token->str
        prn)))
