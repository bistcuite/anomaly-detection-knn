# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# importing data
# url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
data = pd.read_csv("iris.csv")

# input data
df = data[["sepal_length", "sepal_width"]]

# scatterplot of inputs data
plt.scatter(df["sepal_length"], df["sepal_width"])
plt.show()

# create arrays
X = df.values

# instantiate model
nbrs = NearestNeighbors(n_neighbors = 3)

# fit model
nbrs.fit(X)

# distances and indexes of k-neaighbors from model outputs
distances, indexes = nbrs.kneighbors(X)

# plot mean of k-distances of each observation
plt.plot(distances.mean(axis =1))
plt.show()

# visually determine cutoff values > 0.15
outlier_index = np.where(distances.mean(axis = 1) > 0.15)

# filter outlier values
outlier_values = df.iloc[outlier_index]

# plot data
plt.scatter(df["sepal_length"], df["sepal_width"], color = "b", s = 65)
plt.show()

# plot outlier values
plt.scatter(outlier_values["sepal_length"], outlier_values["sepal_width"], color = "r")
plt.show()
