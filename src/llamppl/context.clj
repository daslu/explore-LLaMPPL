(ns llamppl.context
  (:require [tech.v3.datatype :as dtype]
            [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [com.phronemophobic.llama.util :as llutil]
            [tech.v3.datatype.argops :as argops]
            [llamppl.util :as util]
            [clojure.math :as math]))

(defonce llama7b-path "/workspace/models/llama-2-7b-chat.ggmlv3.q4_0.bin")

(def MB (math/pow 2 20))

(defn get-state [llama-ctx]
  (let [size (raw/llama_get_state_size llama-ctx)
        _ (prn [:allocating (format "%.2f" (/ size MB)) "MB"])
        mem (byte-array size)]
    (raw/llama_copy_state_data llama-ctx mem)
    mem))

(defn ->llama-ctx []
  (llama/create-context llama7b-path {}))

(def base-llama-ctx
  (->llama-ctx))

(defn tokenize [text]
  (llutil/tokenize base-llama-ctx text))

(defn untokenize [tokens]
  (llutil/untokenize base-llama-ctx tokens))

(def token->str
  (into (sorted-map)
        (comp (map
               (fn [token]
                 [token (raw/llama_token_to_str base-llama-ctx token)]))
              (take-while (fn [[token untoken]]
                            untoken)))
        (range 0 Integer/MAX_VALUE)))

(def llama-eos (llama/eos base-llama-ctx))

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
  #_(-> context
        (select-keys [:path
                      :tokens
                      :remaining-tokens])
        prn)
  (if (empty? remaining-tokens)
    context
    ;; else
    (let [token (first remaining-tokens)
          rest-tokens (rest remaining-tokens)
          next-step [:children token]
          next-path (concat path next-step)]
      #_(prn [:DBG
              {:path path
               :try (some? (get-in sub-trie next-step))}])
      (if-let [next-sub-trie (get-in sub-trie next-step)]
        (recur (-> context
                   (assoc
                    :sub-trie next-sub-trie
                    :path next-path
                    :remaining-tokens rest-tokens)))
        ;; else
        (let [_ (prn [:EVAL token (untokenize [token])])
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



(delay
  (let [llama-ctx (->llama-ctx)]
    (llama/llama-update llama-ctx (llama/bos) 0)
    (-> {:llama-ctx llama-ctx
         :trie {:llama-state (get-state llama-ctx)}
         :tokens (tokenize "How much wood would a")}
        cached-eval
        :sub-trie
        :logits
        argops/argmax
        token->str)))

(defonce *main-context (atom {}))

(defn init! []
  (let [llama-ctx (->llama-ctx)]
    (llama/llama-update llama-ctx (llama/bos) 0)
    (reset! *main-context
            {:llama-ctx llama-ctx
             :trie {:llama-state (get-state llama-ctx)}}))
  (System/gc))

(init!)



(defn cached-eval! [tokens]
  (let [context (-> @*main-context
                    (assoc :tokens tokens)
                    cached-eval)]
    (reset! *main-context
            (select-keys context [:llama-ctx :trie]))
    context))

(defn logits [tokens]
  (-> tokens
      cached-eval!
      :sub-trie
      :logits))

(delay
  (init!)

  (-> "How much wood would a"
      tokenize
      logits
      argops/argmax
      token->str
      (vector (util/now)))

  (-> "How much wood would a woodchuck"
      tokenize
      logits
      argops/argmax
      token->str
      (vector (util/now)))

  (-> "How much wood would a woodchuck chuck"
      tokenize
      logits
      argops/argmax
      token->str
      (vector (util/now))))


(defn gen-samplef [seed]
  (let [{:keys [llama-ctx]} @*main-context]
    (raw/llama_set_rng_seed llama-ctx seed)
    (llama/init-mirostat-v2-sampler llama-ctx)))

(delay
  (let [samplef (gen-samplef 123456)
        logits (-> "How much wood would a woodchuck chuck"
                   tokenize
                   logits)]
    (untokenize [(samplef logits)])))
