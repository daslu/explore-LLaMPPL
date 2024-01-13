^{:clay {:quarto {:format {:html {:toc true
                                  :toc-depth 4
                                  :theme [:spacelab "notebooks/custom.scss"]}}
                  :highlight-style :solarized
                  :code-block-background true
                  :include-in-header {:text "<link rel = \"icon\" href = \"data:,\" />"}
                  :title "LLaMPPL in a special case: limiting token length"}}}
(ns story1
  (:require [tech.v3.datatype :as dtype]
            [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [com.phronemophobic.llama.util :as llutil]
            [tech.v3.datatype.argops :as argops]
            [clojure.math :as math]
            [clojure.core.cache :as cache]
            [scicloj.kindly.v4.kind :as kind]))

;; This notebook demonstrates a Clojure implementation of a specifica case of LLaMPPL.
;; Specifically, we explore the "Hard Constraints" case from
;; [Sequential Monte Carlo Steering of Large Language Models using Probabilistic Programs](https://arxiv.org/abs/2306.03081)
;; by Alexander K. Lew, Tan Zhi-Xuan, Gabriel Grand, Vikash K. Mansinghka
;; (see Figure 1 and Subsection 2.2).

;; ## Constants

;; Path to the LLM:
(def llama7b-path (str (System/getenv "MODELS_PATH")
                       "/llama-2-7b-chat.ggmlv3.q4_0.bin"))

;; One megabyte:
(def MB (math/pow 2 20))


;; ## Using llama.clj

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

(delay
  (-> "The Fed says"
      tokenize))

;; A function to turn a list of tokens to a String of text:
(defn untokenize [tokens]
  (llutil/untokenize base-llama-ctx tokens))

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

(delay
  (->> "The Fed says"
       tokenize
       (map token->str)))

;; The EOS (end-of-sentence) token:
(def llama-eos (llama/eos base-llama-ctx))

;; Getting next-token logits for a given sequence of tokens.
(delay
  (let [llama-ctx (->llama-ctx)]
    (-> llama-ctx
        (llama/llama-update "What is the")
        llama/get-logits
        vec
        kind/portal)))

;; Getting the mosty probably next-token:
(delay
  (let [llama-ctx (->llama-ctx)]
    (-> llama-ctx
        (llama/llama-update "What is the")
        llama/get-logits
        argops/argmax
        token->str)))

;; Get a copy of a given model's context
;; (so that we can reuse the KV-cache):
(defn get-state [llama-ctx]
  (let [size (raw/llama_get_state_size llama-ctx)
        ;; _ (prn [:allocating (format "%.2f" (/ size MB)) "MB"])
        mem (byte-array size)]
    (raw/llama_copy_state_data llama-ctx mem)
    mem))

(delay
  (let [llama-ctx (->llama-ctx)]
    (-> llama-ctx
        (llama/llama-update "What is the")
        get-state
        count)))
