import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
from decimal import Decimal

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

  # Cluster the data using KMedoids
  kmedoids = KMedoids(n_clusters=num_clusters,max_iter=500, random_state=42)
  kmedoids.fit(vectorized_documents)
  labels = kmedoids.labels_
  medoid_indices = kmedoids.medoid_indices_

  my_obj = {"qid": qid,
            "lables": labels,
            "medoids_indices": medoid_indices
           }

  my_list.append(my_obj)

# Store data (serialize)
with open('Dev/K-medoids/five_clusters.pickle', 'wb') as handle:
    pickle.dump(my_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
