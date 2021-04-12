"""
Shayane Katchera & Valentin Todisco
SET
Ma422 : Introduction to Machine Learning
Research Project about Unsupervised learning

This is an experimentation of a method of unsupervised learning : the K-mean clustering

input data is french airports coordinates
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# initialize variables
k = 5
means = np.array([[random.randint(-6, 11), random.randint(40, 52)] for i in range(k)]) # start of means
limit = 0.01

# extract data
data = []
f = open("data.txt", 'r')
for l in f.readlines():
    line = l.split(',')
    line[-1] = line[-1][:-1]
    data.append([float(line[0]), float(line[1])])
data = np.array(data)

# plot raw data
# plt.scatter(data[:, 0], data[:, 1])
# plt.axis("equal")
# plt.title("Raw data")
# plt.xlabel("latitude")
# plt.ylabel("longitude")
# plt.show()

# usefull functions
def distance(point, cluster):
    """
    Function to compute distance between point and cluster

    parameters:
    point: indexable container (list, tuple) containing integers or float

    cluster: indexable container (list, tuple) containing integers or float
    """
    return np.sqrt((point[0] - cluster[0])**2 + (point[1] - cluster[1])**2)


def StopCondition(old, new, threshold):
    """
    Function that return True if difference between old and new is lower than threshold. 
    Return False if not

    parameters:
    old: list of int/float data type

    new: list of int/float data type

    threshold: int or float
    """
    # check data sizes
    if len(old) != len(new):
        print(f"old and new must be same size. Given {len(old)} and {len(new)}")
        quit()
    
    # compute condition
    for i in range(len(old)):
        if distance(old[i], new[i]) > threshold:
            return True
    return False
    

def GetMiddle(cluster):
    """
    function that return the middle point of the cluster

    parameter
    cluster: list of point
    """
    x = np.mean(np.array([cluster[i][0] for i in range(len(cluster))]))
    y = np.mean(np.array([cluster[i][1] for i in range(len(cluster))]))
    return [x, y]


# main loop
# init counter
s = 0

# reset clusters
clusters = [[] for i in range(k)]

# for each point
for pt in data:
    # compute distance each mean
    dist = [distance(pt, means[i]) for i in range(k)]

    # get closest mean assignment
    j = dist.index(min(dist))
    clusters[j].append(pt)

# set new mean & update counter
newmean = []
for i in range(k):
    newmean.append(GetMiddle(clusters[i]))
s += 1

# test condition and repeat
while StopCondition(means, newmean, limit):
    # reset clusters & update means
    clusters = [[] for i in range(k)]
    means = newmean

    # for each point
    for pt in data:
        # compute distance each mean
        dist = [distance(pt, means[i]) for i in range(k)]

        # get closest mean assignment
        j = dist.index(min(dist))
        clusters[j].append(pt)

    # set new mean sets & update counter
    newmean = []
    for i in range(k):
        newmean.append(GetMiddle(clusters[i]))
    s += 1

# display
for clust in clusters:
    c = np.array(clust)
    try:
        plt.scatter(c[:, 0], c[:, 1], label=f"cluster {clusters.index(clust) + 1}")
    except IndexError:
        pass

# set graph
plt.legend()
plt.title("Clusterd data")
plt.xlabel("latitude")
plt.ylabel("longitude")
plt.show()