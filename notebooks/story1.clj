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
            [tablecloth.api :as tc]
            [clojure.string :as str]))

(def md
  (comp kindly/hide-code kind/md))

(defn now []
  (java.util.Date.))

(defn prn-with-time [x]
  (prn (now) x))

(md
 "**WIP**

This notebook demonstrates a Clojure implementation of a specifica case of LLaMPPL.
Specifically, we explore the \"Hard Constraints\" case from

[Sequential Monte Carlo Steering of Large Language Models
using Probabilistic Programs](https://arxiv.org/abs/2306.03081)

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
  (llama/create-context
   llama7b-path
   {:use-mlock true}))

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

(defn finished? [tokens]
  (-> tokens
      last
      (= llama-eos)))

;; Example:
;; Getting next-token logits for a given sequence of tokens.
(delay
  (-> (->llama-ctx)
      (llama/llama-update "What is the") ; Note this is **mutating** the context
      llama/get-logits
      (->> (take 5))))

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

(def base-state-data
  (ctx->state-data base-llama-ctx))

;; How big is this state?
(delay
  (-> base-state-data
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

;; ## Caching context states:
(def cache-state-data!
  (let [*id (atom 0)]
    (fn [{:keys [state-id
                 state-data-fn
                 *cache]}]
      (let [id (or state-id (swap! *id inc))]
        {:state-id id
         :state-data (cache.wrapped/lookup-or-miss
                      *cache
                      id
                      state-data-fn)}))))

;; Let us try it out with a FIFO cache:
(delay
  (let [*cache (cache.wrapped/fifo-cache-factory
                {}
                {:threshold 60})
        ;; Compute state data and keep it in the cache.
        {:keys [state-id
                state-data]} (cache-state-data!
                              {:*cache *cache
                               :state-data-fn (fn [_]
                                                (text->state-data
                                                 "What is the"))})
        ;; Use the cache a few more times.
        _ (dotimes [i 3]
            (cache-state-data!
             {:*cache *cache
              :state-data-fn (fn [_]
                               (text->state-data
                                (str "What is " i)))}))
        ;; Retrieve our state data from the first call.
        retrieved-state-data (-> {:*cache *cache
                                  :state-id state-id}
                                 cache-state-data!
                                 :state-data)]
    ;; Make sure it is equals.
    (java.util.Arrays/equals
     ^bytes state-data
     ^bytes retrieved-state-data)))


;; ## A token-trie cache
;; (Section 3, Subsection "Shared Transformer cache" in the paper)



(defn cached-eval [{:as context
                    :keys [llama-ctx
                           llama-ctx-state-id
                           trie
                           *cache
                           tokens
                           path
                           sub-trie
                           remaining-tokens
                           num-threads-to-use]
                    :or {sub-trie trie
                         path []
                         remaining-tokens tokens
                         num-threads-to-use (/ *num-threads* 2)}}]
  (if (empty? remaining-tokens)
    ;; done - return this context
    context
    ;; else
    (let [token (first remaining-tokens)
          ;; Look into the next sub trie,
          ;; following this token:
          next-step [:children token]
          next-path (concat path next-step)
          next-sub-trie (get-in sub-trie next-step)]
      (if (some->> next-sub-trie
                   :llama-state-id
                   (cache.wrapped/has? *cache))
        ;; We have already created next-sub-trie in the past,
        ;; and we have its llama state still in the cache,
        ;; so let us step into it.
        (recur (-> context
                   (assoc
                    :sub-trie next-sub-trie
                    :path next-path
                    :remaining-tokens (rest remaining-tokens))))
        ;; Else, we need to create the next sub trie.
        (let [{:keys [state-id state-data]}
              (cache-state-data!
               {:*cache *cache
                :state-data-fn (fn [_]

                                 ;; Make sure the llama-ctx has the right state
                                 ;; to continue.
                                 (cond
                                   ;; When we are in the beginning of the path,
                                   ;; take the base state.
                                   (= path [])
                                   (raw/llama_set_state_data llama-ctx
                                                             base-state-data)
                                   ;; When the last evaluation does not fit
                                   ;; out place in the trie,
                                   ;; bring the reletant state from cache.
                                   (-> sub-trie
                                       :llama-state-id
                                       (not= llama-ctx-state-id))
                                   (->> sub-trie
                                        :llama-state-id
                                        (cache.wrapped/lookup *cache)
                                        (raw/llama_set_state_data llama-ctx))
                                   ;; Otherwise, our current state is what we need.
                                   :else
                                   nil)
                                 ;; Evaluate the current token:
                                 (prn [:eval
                                       (->> path
                                            (filter number?)
                                            untokenize)
                                       '-->
                                       (untokenize [token])])
                                 (time
                                  (llama/llama-update llama-ctx
                                                      token
                                                      num-threads-to-use))
                                 (ctx->state-data llama-ctx))})
              ;; Create the next sub trie:
              new-sub-trie {:logits (llama/get-logits llama-ctx)
                            :llama-state-id state-id}]
          ;; Step into the next sub trie:
          (recur (-> context
                     (update :trie assoc-in next-path new-sub-trie)
                     (assoc :llama-ctx-state-id state-id
                            :sub-trie new-sub-trie
                            :path next-path
                            :remaining-tokens (rest remaining-tokens)))))))))

(defn new-context
  ([]
   (new-context {}))
  ([{:keys [lru-params seed]
     :or {lru-params {:threshold 20}
          seed 12345}}]
   (System/gc)
   (let [llama-ctx (->llama-ctx)
         samplef (llama/init-mirostat-v2-sampler
                  llama-ctx)]
     (prn [:seed seed])
     (raw/llama_set_rng_seed llama-ctx seed)
     {:llama-ctx llama-ctx
      :samplef samplef
      :trie {}
      :*cache (cache.wrapped/lru-cache-factory
               {}
               lru-params)})))

(defn cached-eval! [*context tokens]
  (let [context (-> @*context
                    (assoc :tokens tokens)
                    cached-eval)]
    (reset! *context
            (select-keys context [:llama-ctx :*cache :samplef :trie]))
    context))

(defn logits! [*context tokens]
  (-> *context
      (cached-eval! tokens)
      :sub-trie
      :logits))


(delay
  (let [*context (atom (new-context))]
    (->> "How much wood would a"
         tokenize
         (logits! *context)
         argops/argmax
         token->str)))


(delay
  (let [*context (atom (new-context))]
    (->> ["How much wood would a"
          "How much wood would a woodchuck"
          "How much wood would a woodchuck chuck"]
         (mapv (fn [text]
                 (->> text
                      tokenize
                      (logits! *context)
                      argops/argmax
                      token->str
                      (vector (now))))))))

;; ## Sampling random tokens

(delay
  (let [*context (atom (new-context {:seed 1}))
        {:keys [samplef]} @*context]
    (->> ["How much wood would a"
          "How much wood would a woodchuck"
          "How much wood would a woodchuck chuck"]
         (mapv
          (fn [text]
            (let [logits (->> text
                              tokenize
                              (logits! *context))]
              [text
               (->> (repeatedly
                     1000
                     #(untokenize [(samplef logits)]))
                    frequencies)]))))))


(defn sample-once! [*context logits]
  (-> @*context
      :llama-ctx
      (raw/llama_set_rng_seed (rand-int 9999)))
  ((:samplef @*context)
   logits))

(delay
  (let [*context (atom (new-context))]
    (->> "How much wood would a"
         tokenize
         (logits! *context)
         (sample-once! *context)
         vector
         untokenize)))

;; ## A probabilistic model

;; The LLaMPPL paper defines a probabilistic model using
;; an initial state $s_$,
;; a Markove kernel $M$,
;; and a potential function $G$.
;;
;; Here we are implmenting a specific model
;; of the 'hard constraints' type:
;; generating texts that use only short words.
;;
;; ### The Markov kernel
;; We define M as a sampling step
;; (which is the way we use it algorithmically).

(defn M-step [*context
              previous-tokens]
  (->> previous-tokens
       (logits! *context)
       (sample-once! *context)
       (conj previous-tokens)))


(delay
  (let [*context (atom (new-context {:seed 1}))]
    (->> #(->> "How much wood"
               tokenize
               (iterate (partial M-step *context))
               (take 5)
               last
               untokenize)
         (repeatedly 5)
         vec)))


(delay
  (let [*context (atom (new-context {:seed 1}))]
    (->> #(->> "The Fed says"
               tokenize
               (iterate (partial M-step *context))
               (take 30)
               last
               ((juxt finished?
                      untokenize)))
         (repeatedly 2)
         vec)))

;; ### The potential function



#_(defn G [threshold current-tokens]
    (if (-> current-tokens
            context/untokenize
            (str/split  #" ")
            (->> (every? #(-> % count (<= threshold)))))
      1 0))
