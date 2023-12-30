(ns llamppl.util)

(defn now []
  (java.util.Date.))

(defn spy [x tag]
  (prn [tag x])
  x)
