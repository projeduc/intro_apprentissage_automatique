#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import matplotlib.pyplot as plt
import math
import csv
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X = numpy.arange(0, 4, 0.25)


func = lambda x: x + math.cos(x * math.pi)

vfunc = numpy.vectorize(func)

Y = vfunc(X)

Xb = X.reshape(-1, 1)

Yb = Y.reshape(-1, 1)

#print Xb

reg = LinearRegression().fit(Xb, Yb)

Y_lin = reg.predict(Xb)

plt.plot(X, Y, "o-")
plt.plot(X, Y_lin, "v-")

plt.legend(["originale", u"estimée"])

plt.xlabel("x")
plt.ylabel("y")
plt.title(u"Régression linéaire")
plt.grid()
plt.show()

poly = PolynomialFeatures(degree=4)
X_ = poly.fit_transform(Xb)

#reg = IsotonicRegression().fit(X, Y)
reg = LinearRegression().fit(X_, Yb)
Y_lin = reg.predict(X_)
plt.plot(X, Y, "o-")
plt.plot(X, Y_lin, "v-")

plt.legend(["originale", u"estimée"])

plt.xlabel("x")
plt.ylabel("y")
plt.title(u"Régression polynomiale (degrée 4)")
plt.grid()
plt.show()

#df = DataFrame({"x": X, "y": Y})
#df.to_csv("res.csv", index=False)
