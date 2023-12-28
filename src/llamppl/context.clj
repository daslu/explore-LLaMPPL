(ns llamppl.context
  (:require [tech.v3.datatype :as dtype]
            [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]))

(defonce llama7b-path "/workspace/models/llama-2-7b-chat.ggmlv3.q4_0.bin")

(defn llama-copy [ctx]
  (let [mem (byte-array (raw/llama_get_state_size ctx))]
    (raw/llama_copy_state_data ctx mem)
    mem))

(defn ->token-trie [details]
  (-> details
      (assoc :children {})))

(defn get-token [token-trie token-id]
  (-> token-trie
      :children
      token-id))

(defn add-token [token-trie token-id details]
  (-> token-trie
      (assoc-in [:children token-id]
                (-> details
                    (assoc :parent token-trie)))))

(defn check-cache-size [kv-index
                        n-new
                        llama-ctx]
  (let [n-ctx (raw/llama_n_ctx llama-ctx)]
    (when (-> kv-index
              (+ n-new)
              (>= n-ctx))
      (throw (ex-info "cache overflows the context size"
                      {:kv-index kv-index
                       :n-new n-new
                       :n-ctx n-ctx})))))

;; (defn cached-eval [tokens {:keys [llama-ctx
;;                                   token-trie
;;                                   logprob]}]
;;   (let [n-new (count tokens)
;;         {:keys [kv-index]} token-trie]
;;     (check-cache-size kv-index n-new llama-ctx)))


;; (defn active-llama []
;;   (-> {:kv-index -1
;;        :trie (->token-trie {:kv-index 0})}
;;       (cached-eval [llama/bos])))
