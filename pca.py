import numpy as np
from sklearn.decomposition import PCA
import numpy
from numpy import genfromtxt
import csv

data = genfromtxt('data.csv',delimiter=',')
'''target = []

for ex in data:
    target.append(ex[37])
TARGET =  numpy.array(target)'''

pca = PCA(n_components=37)
pca.fit_transform(data)
print(sum(PCA(n_components = 37)))
#pca.get_precision(TARGET)
#print(pca.explained_variance_ratio_)
print(pca.singular_values_)