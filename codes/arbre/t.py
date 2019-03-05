"""
==========================
Graph export from Estimator
==========================
An example graph export of :class:`id3.id3.Id3Estimator` with
:file:`id3.export.export_graphviz`
$ dot -T png out.dot -o out.png
.. figure::  /_static/out.png
   :align:   center
"""

from id3 import Id3Estimator, export_graphviz
import numpy as np

feature_names = [
                 "gender",
                 "sector",
                 "degree"]

X = np.array([["male", "private", "m"],
              ["female", "private", "m"],
              ["other", "public", "b"],
              ["male", "private", "none"],
              ["female", "private", "none"],
              ["male", "public", "none"],
              ["other", "private", "m"],
              ["male", "private", "m"],
              ["female", "private", "m"],
              ["male", "public", "m"],
              ["other", "public", "m"],
              ["other", "public", "b"],
              ["female", "public", "b"],
              ["male", "public", "b"],
              ["female", "private", "b"],
              ["male", "private", "b"],
              ["other", "private", "b"]])

y = np.array(["(30k,38k)",
              "(30k,38k)",
              "(30k,38k)",
              "(13k,15k)",
              "(13k,15k)",
              "(13k,15k)",
              "(23k,30k)",
              "(23k,30k)",
              "(23k,30k)",
              "(15k,23k)",
              "(15k,23k)",
              "(15k,23k)",
              "(15k,23k)",
              "(15k,23k)",
              "(23k,30k)",
              "(23k,30k)",
              "(23k,30k)"])

print X

clf = Id3Estimator()

#X = X.astype(np.float64)
clf.fit(X, y, check_input=True)

export_graphviz(clf.tree_, "out.dot", feature_names)
