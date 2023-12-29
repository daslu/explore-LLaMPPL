(ns llamppl.context
  (:require [tech.v3.datatype :as dtype]
            [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [com.phronemophobic.llama.util :as llutil]
            [tech.v3.datatype.argops :as argops]))

(defonce llama7b-path "/workspace/models/llama-2-7b-chat.ggmlv3.q4_0.bin")

(defn get-state [llama-ctx]
  (let [mem (byte-array (raw/llama_get_state_size llama-ctx))]
    (raw/llama_copy_state_data llama-ctx mem)
    mem))

(defn ->llama-ctx []
  (llama/create-context llama7b-path {}))

(def token->str
  (let [llama-ctx (->llama-ctx)]
    (into (sorted-map)
          (comp (map
                 (fn [token]
                   [token (raw/llama_token_to_str llama-ctx token)]))
                (take-while (fn [[token untoken]]
                              untoken)))
          (range 0 Integer/MAX_VALUE))))

(defn ->token-trie [details]
  (-> details
      (assoc :children {})))

(defn get-token [trie token]
  (-> trie
      :children
      token))

(defn add-token [trie token details]
  (-> trie
      (assoc-in [:children token]
                (-> details
                    (assoc :parent trie)))))

(defn check-cache-size [kv-index
                        n-new
                        llama-ctx]
  (let [n-ctx (raw/llama_n_ctx llama-ctx)]
    (when (-> kv-index
              (+ n-new)
              (>= n-ctx))
      (throw (ex-info "cache overflows the llama-ctx size"
                      {:kv-index kv-index
                       :n-new n-new
                       :n-ctx n-ctx})))))


(defn cached-eval [{:as context
                    :keys [llama-ctx
                           trie
                           sub-trie
                           path
                           tokens
                           remaining-tokens]
                    :or {sub-trie trie
                         path []
                         remaining-tokens tokens}}]
  (if (empty? remaining-tokens)
    context
    ;; else
    (let [token (first remaining-tokens)
          rest-tokens (rest remaining-tokens)
          next-step [:children token]
          next-path (concat path next-step)]
      (if-let [next-sub-trie (get-in sub-trie next-step)]
        (recur (-> context
                   (assoc
                    :sub-trie next-sub-trie
                    :path next-path
                    :remaining-tokens rest-tokens)))
        ;; else
        (let [_ (prn token)
              _ (raw/llama_set_state_data llama-ctx
                                          (:llama-state sub-trie))
              _ (llama/llama-update llama-ctx
                                    token)
              new-sub-trie {:logits (llama/get-logits llama-ctx)
                            :llama-state (get-state llama-ctx)}]
          (recur (-> context
                     (update :trie assoc-in next-path new-sub-trie)
                     (assoc :sub-trie new-sub-trie
                            :path next-path
                            :remaining-tokens rest-tokens))))))))

(comment
  (let [llama-ctx (->llama-ctx)]
    (llama/llama-update llama-ctx (llama/bos) 0)
    (-> {:llama-ctx llama-ctx
         :trie {:llama-state (get-state llama-ctx)}
         :tokens (->> "How much wood would a"
                      (llutil/tokenize llama-ctx))}
        cached-eval
        :sub-trie
        :logits
        argops/argmax
        token->str)))
