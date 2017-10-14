import keras
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import MiniBatchKMeans 
dataset = pd.read_csv('/home/saranshk2/Downloads/mnist_train.csv')
test=pd.read_csv('/home/saranshk2/Downloads/mnist_test.csv')
x = dataset.iloc[0:, 1:].values
x=scale(x)
X=np.array(x)
y = dataset.iloc[0:,0].values
x_test=test.iloc[0:, 1:].values
x_test=scale(x_test)
y_test=test.iloc[0:, 0].values
reduced_data = PCA(n_components=2).fit_transform(x)
reduced_data=np.array(reduced_data)
reduced_test_data=PCA(n_components=2).fit_transform(x_test)
reduced_test_data=np.array(reduced_test_data)
kmeans=MiniBatchKMeans(n_clusters=10, init='k-means++', max_iter=100, batch_size=100, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01)
kmeans.fit(reduced_data)
centroid=kmeans.cluster_centers_
labels=kmeans.labels_
print reduced_data.shape
plt.scatter(reduced_data[:,0],reduced_data[:,1],c=[matplotlib.cm.spectral(float(i) /10) for i in labels])
plt.scatter(centroid[:,0],centroid[:,1],color="g",marker="x")
plt.show()


