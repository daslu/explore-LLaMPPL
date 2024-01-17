^{:clay {:quarto {:title "LLaMPPL in a special case: limiting token length"}}}
(ns story1
  (:require [tech.v3.datatype :as dtype]
            [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [com.phronemophobic.llama.util :as llutil]
            [tech.v3.datatype.argops :as argops]
            [clojure.math :as math]
            [clojure.core.cache :as cache]
            [clojure.core.cache.wrapped :as cache.wrapped]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.kindly.v4.api :as kindly]
            [scicloj.noj.v1.vis.hanami :as hanami]
            [aerial.hanami.templates :as ht]
            [tablecloth.api :as tc]))

(def md
  (comp kindly/hide-code kind/md))

(md
 "**WIP**

This notebook demonstrates a Clojure implementation of a specifica case of LLaMPPL.
Specifically, we explore the \"Hard Constraints\" case from
[Sequential Monte Carlo Steering of Large Language Models using Probabilistic Programs](https://arxiv.org/abs/2306.03081)
by Alexander K. Lew, Tan Zhi-Xuan, Gabriel Grand, Vikash K. Mansinghka
(see Figure 1 and Subsection 2.2).")

;; ## Constants

;; Path to the LLM:
(def llama7b-path (str (System/getenv "MODELS_PATH")
                       "/llama-2-7b-chat.ggmlv3.q4_0.bin"))

;; One megabyte:
(def MB (math/pow 2 20))


;; ## Using llama.clj

(md "We will use [llama.clj](https://github.com/phronmophobic/llama.clj), a Clojure wrapper of llama.cpp.")

;; Create a new model context:
(defn ->llama-ctx []
  (llama/create-context llama7b-path {}))

;; A copy of an empty model
;; (to extract basic information):
(def base-llama-ctx
  (->llama-ctx))

;; A function to turn a String of text to a list of tokens:
(defn tokenize [text]
  (llutil/tokenize base-llama-ctx text))

;; Example:
(delay
  (-> "The Fed says"
      tokenize))

;; A function to turn a list of tokens to a String of text:
(defn untokenize [tokens]
  (llutil/untokenize base-llama-ctx tokens))

;; Example:
(delay
  (-> "The Fed says"
      tokenize
      untokenize))

;; A map from tokens to the corresponding strings:
(def token->str
  (into (sorted-map)
        (comp (map
               (fn [token]
                 [token (raw/llama_token_to_str base-llama-ctx token)]))
              (take-while (fn [[token untoken]]
                            untoken)))
        (range 0 Integer/MAX_VALUE)))

;; Example:
(delay
  (->> "The Fed says"
       tokenize
       (map token->str)))

;; The EOS (end-of-sentence) token:
(def llama-eos (llama/eos base-llama-ctx))

;; Example:
;; Getting next-token logits for a given sequence of tokens.
(delay
  (-> (->llama-ctx)
      (llama/llama-update "What is the") ; Note this is **mutating** the context
      llama/get-logits
      vec
      kind/portal))

(delay
  (-> (->llama-ctx)
      (llama/llama-update "What is the")
      llama/get-logits
      (->> (hash-map :logit))
      tc/dataset
      (hanami/histogram :logit {:nbins 100})
      (assoc :height 200)))

;; Example:
;; Getting the most probable next-token:
(delay
  (let [llama-ctx (->llama-ctx)]
    (-> llama-ctx
        (llama/llama-update "What is the")
        llama/get-logits
        argops/argmax
        token->str)))

;; ## Keeping and retrieving context state

;; A function to get a copy of a given model-context's state
;; (so that we can reuse the KV-cache later):
(defn ctx->state-data [llama-ctx]
  (let [size (raw/llama_get_state_size llama-ctx)
        mem (byte-array size)]
    (raw/llama_copy_state_data llama-ctx mem)
    mem))

;; How big is this state?
(delay
  (-> (->llama-ctx)
      (llama/llama-update "What is the")
      ctx->state-data
      count
      (/ MB)
      (->> (format "%.02f MB"))))

;; A function to recreate a model-cotext from its state:
(defn state-data->ctx [state-data]
  (let [llama-ctx (->llama-ctx)]
    (raw/llama_set_state_data llama-ctx state-data)
    llama-ctx))

;; A function to turn a piece of text to the corresponding
;; model state data:
(defn text->state-data [text]
  (-> (->llama-ctx)
      (llama/llama-update text)
      ctx->state-data))

;; Example:
;; Keep, retrieve, and reuse the state
(delay
  (-> "What is the"
      text->state-data
      state-data->ctx
      llama/get-logits
      argops/argmax
      token->str))

;; ## An LRU cache for context states
(def *id
  (atom 0))

(defn next-id! []
  (swap! *id inc))

(def *state-data-cache
  (cache.wrapped/lru-cache-factory
   {}
   {:threshold 5}))

(defn cache-state-data! [{:keys [state-data-id state-data-fn]
                          :or {state-data-fn (next-id!)}}]
  {:state-data-id state-data-id
   :state-data (cache.wrapped/lookup-or-miss
                *state-data-cache
                state-data-id
                state-data-fn)})

;; Let us try it out:
(delay
  (let [{:keys [state-data-id
                state-data]} (cache-state-data!
                              {:state-data-fn (fn [_]
                                                (text->state-data
                                                 "What is the"))})
        retrieved-state-data (-> {:state-data-id state-data-id}
                                 cache-state-data!
                                 :state-data)]
    (java.util.Arrays/equals
     ^bytes state-data
     ^bytes retrieved-state-data)))


;; ## A token-trie cache
;; (Section 3, Subsection "Shared Transformer cache" in the paper)
