^{:clay {:quarto {:title "LLaMPPL in a special case: limiting token length"}}}
(ns story1
  (:require [tech.v3.datatype :as dtype]
            [com.phronemophobic.llama :as llama]
            [com.phronemophobic.llama.raw :as raw]
            [com.phronemophobic.llama.util :as llutil]
            [tech.v3.datatype.argops :as argops]
            [clojure.math :as math]
            ;; [clojure.core.cache :as cache]
            ;; [clojure.core.cache.wrapped :as cache.wrapped]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.kindly.v4.api :as kindly]
            [scicloj.noj.v1.vis.hanami :as hanami]
            [aerial.hanami.templates :as ht]
            [tablecloth.api :as tc]
            [clojure.string :as str]
            [clojure.walk :as walk]))

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
(defn new-llama-ctx []
  (llama/create-context
   llama7b-path
   {:use-mlock true}))



;; A copy of an empty model
;; (to extract basic information):
(def base-llama-ctx
  (new-llama-ctx))

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



;; The EOS (end-of-sentence) token:Color
(def llama-eos (llama/eos base-llama-ctx))

(defn finished? [tokens]
  (-> tokens
      last
      (= llama-eos)))

;; Example:
;; Getting next-token logits for a given sequence of tokens.
(delay
  (-> (new-llama-ctx)
      (llama/llama-update "What is the") ; Note this is **mutating** the context
      llama/get-logits
      (->> (take 5))))

(delay
  (-> (new-llama-ctx)
      (llama/llama-update "What is the")
      llama/get-logits
      (->> (hash-map :logit))
      tc/dataset
      (hanami/histogram :logit {:nbins 100})
      (assoc :height 200)))

;; Example:
;; Getting the most probable next-token:
(delay
  (let [llama-ctx (new-llama-ctx)]
    (-> llama-ctx
        (llama/llama-update "What is the")
        llama/get-logits
        argops/argmax
        token->str)))

(delay
  (let [llama-ctx (new-llama-ctx)]
    (-> llama-ctx
        (llama/llama-update "How much wood would a")
        llama/get-logits
        argops/argmax
        token->str)))

;; ## Keeping and retrieving context state

(def state-size
  (-> base-llama-ctx
      (raw/llama_get_state_size)))

;; How big is this state?
(delay
  (-> state-size
      (/ MB)
      (->> (format "%.02f MB"))))

(def base-state-data
  (let [mem (byte-array state-size)]
    (raw/llama_copy_state_data base-llama-ctx mem)
    mem))

;; Let us create a space to store a few such states:
(def n-memories 10)

(defonce memories
  (vec (repeatedly n-memories #(byte-array state-size))))

(delay
  (->> memories
       (map count)
       frequencies))


(delay
  (let [llama-ctx (new-llama-ctx)
        next-word (fn []
                    (-> llama-ctx
                        llama/get-logits
                        argops/argmax
                        vector
                        untokenize))]
    (-> llama-ctx
        (llama/llama-update "How much wood")
        time)
    (-> llama-ctx
        (raw/llama_copy_state_data (memories 3))
        time)
    (prn (next-word))
    (-> llama-ctx
        (llama/llama-update "How are you")
        time)
    (prn (next-word))
    (-> llama-ctx
        (raw/llama_set_state_data (memories 3))
        time)
    (prn (next-word))))


(defn new-fifo-cache []
  {:id->idx {}
   :idx->id {}
   :current-idx 0})


(defn lookup-or-miss [*fifo-cache id mem-cpy-fn]
  (let [{:keys [id->idx]} @*fifo-cache]
    (or (some-> id
                id->idx
                memories)
        (-> *fifo-cache
            (swap! (fn [fifo-cache]
                     (let [updated-fifo-cache
                           (as-> fifo-cache fc
                             (update fc :current-idx
                                     (fn [idx] (-> idx inc (rem n-memories))))
                             (update fc :id->idx dissoc ((:idx->id fc)
                                                         (:current-idx fc)))
                             (update fc :id->idx assoc id (:current-idx fc))
                             (update fc :idx->id assoc (:current-idx fc) id))]
                       (-> updated-fifo-cache
                           :current-idx
                           memories
                           mem-cpy-fn)
                       updated-fifo-cache)))
            :current-idx
            memories))))

(defn has? [*fifo-cache id]
  (-> @*fifo-cache
      :id->idx
      (contains? id)))

(defn lookup [*fifo-cache id]
  (prn [@*fifo-cache
        id])
  (-> @*fifo-cache
      :id->idx
      (get id)
      memories))

(def cache-state-data!
  (let [*id (atom 0)]
    (fn [{:keys [state-id
                 llama-ctx-fn
                 *cache]}]
      (let [id (or state-id (swap! *id inc))]
        {:state-id id
         :state-data (lookup-or-miss
                      *cache
                      id
                      (fn [mem]
                        (raw/llama_copy_state_data
                         (llama-ctx-fn)
                         mem)))}))))


(delay
  (let [llama-ctx (new-llama-ctx)
        next-word (fn []
                    (-> llama-ctx
                        llama/get-logits
                        argops/argmax
                        vector
                        untokenize))
        *cache (atom (new-fifo-cache))
        _ (dotimes [i 15]
            (cache-state-data!
             {:*cache *cache
              :llama-ctx-fn #(llama/llama-update
                              llama-ctx
                              "How are you")})
            (prn (next-word)))
        {:keys [state-id
                state-data]} (cache-state-data!
                              {:*cache *cache
                               :llama-ctx-fn #(llama/llama-update
                                               llama-ctx
                                               "How much wood")})]
    (prn (next-word))
    (dotimes [i 3]
      (cache-state-data!
       {:*cache *cache
        :llama-ctx-fn #(llama/llama-update
                        llama-ctx
                        "How are you")})
      (prn (next-word)))
    (let [retrieved-state-data (lookup *cache state-id)]
      (prn
       (java.util.Arrays/equals
        ^bytes state-data
        ^bytes retrieved-state-data))
      (-> llama-ctx
          (raw/llama_set_state_data
           retrieved-state-data))
      (prn (next-word)))))





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
                           remaining-tokens]
                    :or {sub-trie trie
                         path []
                         remaining-tokens tokens}}]
  (let [path->text (fn [path]
                     (->> path
                          (filter number?)
                          untokenize))]
    (if (empty? remaining-tokens)
      ;; done - return this context
      (do
        (prn [:done (path->text path)])
        context)
      ;; else
      (let [token (first remaining-tokens)
            ;; Look into the next sub trie,
            ;; following this token:
            next-step [:children token]
            next-path (concat path next-step)
            next-sub-trie (get-in sub-trie next-step)]
        (if (some->> next-sub-trie
                     :llama-state-id
                     (has? *cache))
          ;; We have already created next-sub-trie in the past,
          ;; and we have its llama state still in the cache,
          ;; so let us step into it.
          (do
            (prn [:recur-known (path->text next-path)])
            (recur (-> context
                       (assoc
                        :sub-trie next-sub-trie
                        :path next-path
                        :remaining-tokens (rest remaining-tokens)))))
          ;; Else, we need to create the next sub trie.
          (let [{:keys [state-id state-data]}
                (cache-state-data!
                 {:*cache *cache
                  :llama-ctx-fn (fn []
                                  ;; Make sure the llama-ctx has the right state
                                  ;; to continue.
                                  (cond
                                    ;; When we are in the beginning of the path,
                                    ;; take the base state.
                                    (= path [])
                                    (do
                                      (prn [:set-from-base])
                                      (raw/llama_set_state_data llama-ctx
                                                                base-state-data))
                                    ;; When the last evaluation does not fit
                                    ;; out place in the trie,
                                    ;; bring the reletant state from cache.
                                    (-> sub-trie
                                        :llama-state-id
                                        (not= llama-ctx-state-id))
                                    (do
                                      (prn [:set-from-cache])
                                      (->> sub-trie
                                           :llama-state-id
                                           (lookup *cache)
                                           (raw/llama_set_state_data llama-ctx)))
                                    ;; Otherwise, our current state is what we need.
                                    :else
                                    (prn [:continue]))
                                  ;; Evaluate the current token:
                                  (prn [:eval
                                        (path->text path)
                                        '-->
                                        (untokenize [token])])
                                  (time
                                   (llama/llama-update llama-ctx
                                                       token
                                                       ;; num of threads
                                                       8))
                                  (prn [:extract-state])
                                  llama-ctx)})
                ;; Create the next sub trie:
                new-sub-trie {:logits (llama/get-logits llama-ctx)
                              :llama-state-id state-id}]
            ;; Step into the next sub trie:
            (do (prn [:recur-new (path->text next-path)])
                (recur (-> context
                           (update :trie assoc-in next-path new-sub-trie)
                           (assoc :llama-ctx-state-id state-id
                                  :sub-trie new-sub-trie
                                  :path next-path
                                  :remaining-tokens (rest remaining-tokens)))))))))))


(defn new-context
  ([]
   (new-context {}))
  ([{:keys [seed]
     :or {seed 12345}}]
   (System/gc)
   (let [llama-ctx (new-llama-ctx)
         samplef (llama/init-mirostat-v2-sampler
                  llama-ctx)]
     (prn [:seed seed])
     (raw/llama_set_rng_seed llama-ctx seed)
     {:llama-ctx llama-ctx
      :samplef samplef
      :trie {}
      :*cache (atom (new-fifo-cache))})))

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

(def example-context
  (let [*context (atom (new-context))]
    (->> "How much wood would a"
         tokenize
         (cached-eval! *context))
    (->> "How much is it"
         tokenize
         (cached-eval! *context))
    (->> "How are"
         tokenize
         (cached-eval! *context))
    @*context))


(let [{:keys [*cache trie]} example-context
      *node-id (atom 0)
      *nodes (atom [{:data {:id "0" :word "(root)"}}])
      *edges (atom [])
      trie (-> trie
               (assoc :node-id (str @*node-id))
               (->> (walk/prewalk
                     (fn [v]
                       (if (:children v)
                         (-> v
                             (update :children
                                     (fn [children]
                                       (->> children
                                            (map (fn [[token child]]
                                                   (let [node-id (str (swap! *node-id inc))]
                                                     (swap! *nodes conj {:data {:id node-id
                                                                                :token token
                                                                                :word (untokenize [token])
                                                                                :background (if (->> child
                                                                                                     :llama-state-id
                                                                                                     (has? *cache))
                                                                                              "lightgreen"
                                                                                              "lightgrey")}})
                                                     [token (-> child
                                                                (assoc :node-id node-id))])))
                                            (into {})))))
                         v)))
                    (walk/prewalk (fn [v]
                                    (if (:logits v)
                                      (dissoc v :logits)
                                      v)))
                    (walk/prewalk
                     (fn [v]
                       (if-let [{:keys [node-id]} v]
                         (do
                           (->> v
                                :children
                                vals
                                (map
                                 (fn [child]
                                   (let [child-node-id (:node-id child)]
                                     {:data {:id (str node-id "-" child-node-id)
                                             :source node-id
                                             :target child-node-id}})))
                                (swap! *edges concat))
                           v)
                         v)))))]
  (kind/cytoscape
   { ;:trie trie
    :elements {:nodes @*nodes
               :edges @*edges}
    :style [{:selector "node"
             :css {:content "data(word)"
                   :text-valign "center"
                   :text-halign "center"
                   :height 50
                   :width 50
                   :background-color "data(background)"}}
            {:selector "edge"
             :css {:curve-style "bezier"
                   :target-arrow-shape "triangle"}}]
    :layout {:name "cose"}}))


(delay
  (let [*context (atom (new-context))]
    (->> ["How"
          "How much"
          "How much wood"
          "How much wood would"
          "How much wood would a"]
         (mapv (fn [prefix]
                 [prefix
                  (->> prefix
                       tokenize
                       (logits! *context)
                       argops/argmax
                       token->str)])))))


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
    (->> #(->> "How much wood"
               tokenize
               (iterate (partial M-step *context))
               (take 40)
               last
               untokenize)
         (repeatedly 2)
         vec)))

(delay
  (let [*context (atom (new-context {:seed 1}))]
    (->> #(->> "The Fed says"
               tokenize
               (iterate (partial M-step *context))
               (take 40)
               (mapv (juxt finished?
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
