import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from decimal import Decimal
from scipy.spatial.distance import euclidean

my_list = list()

for i,row in dev_queries.iterrows():
  qid = row['id']
  retrieved_list = all_result.loc[all_result['qid'] == qid]
  documents = retrieved_list["passage"]

  # create vectorizer
  vectorizer = TfidfVectorizer(stop_words='english')

  # vectorizer the text documents
  vectorized_documents = vectorizer.fit_transform(documents)
  matrix = vectorized_documents.toarray()

  # reduce the dimensionality of the data using PCA
  pca = PCA(n_components=2)
  reduced_data = pca.fit_transform(vectorized_documents.toarray())

  num_clusters = 5
    
  # Cluster the documents using K-Means
  kmeans = KMeans(n_clusters=num_clusters, n_init="auto",max_iter=500, random_state=42)
  kmeans.fit(vectorized_documents)
  labels = kmeans.labels_  

# Loop over all clusters and find index of closest point to the cluster center and append to closest_pt_idx list.
  closest_pt_idx = []
  for iclust in range(kmeans.n_clusters):
      # get all points assigned to each cluster:
      cluster_pts = matrix[kmeans.labels_ == iclust]
   
      # get all indices of points assigned to this cluster:
      cluster_pts_indices = np.where(kmeans.labels_ == iclust)[0]

      cluster_cen = kmeans.cluster_centers_[iclust]
      min_idx = np.argmin([euclidean(matrix[idx], cluster_cen) for idx in cluster_pts_indices])
      closest_pt_idx.append(cluster_pts_indices[min_idx])

  my_obj = {"qid": qid,
            "lables": labels,
            "medoids_indices": closest_pt_idx
           }

  my_list.append(my_obj)

# Store data (serialize)
with open('Dev/K-means/five_clusters.pickle', 'wb') as handle:
    pickle.dump(my_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
