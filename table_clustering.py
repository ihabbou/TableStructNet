import numpy as np

from sklearn.cluster import DBSCAN, MeanShift
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import image_utils

def _tri_coordinates_x(words):
    """ gives us a list of ocr points [[left, center, right], ...]"""
    return [(cord[0], center[0], cord[2])
         for center, cord in zip(
             image_utils.get_centroids_from_coords([word['coords'] for word in words]), 
             [word['coords'] for word in words]
         )
        ]

def _tri_coordinates_y(words):
    """ gives us a list of ocr points [[top, center, bottom], ...]"""
    return [(cord[1], center[1], cord[3])
         for center, cord in zip(
             image_utils.get_centroids_from_coords([word['coords'] for word in words]), 
             [word['coords'] for word in words]
         )
        ]

def cluster_column_tri(words, table_bounds, scale=True, triple=True):

    Xx = _tri_coordinates_x(words)
    Xy = _tri_coordinates_y(words)

    Xx = [x for xl in Xx for x in xl] # flatten
    Xy = [x for xl in Xy for x in xl] # flatten

    print(table_bounds)
    # print(cents)
    if scale:
        scaled = image_utils.scale_centroids(zip(Xx, Xy), table_bounds)
        
        Xxs = [c[0] for c in scaled]
        Xys = [c[1] for c in scaled]

        Xxs = np.array(Xxs).reshape(-1, 1)
        Xys = np.array(Xys).reshape(-1, 1)
    else: 
        Xxs = np.array(Xx).reshape(-1, 1)
        Xys = np.array(Xy).reshape(-1, 1)

    # print(X)

    # #############################################################################
    # Compute DBSCAN
    Xxs = Xxs if triple \
        else [t[1] for t in [Xxs[i:i+3] for i in range(0, len(Xxs), 3)]]
    
    print(sorted([list(x)[0] for x in Xxs]))
    dbx = DBSCAN(eps=0.02, min_samples=1).fit(Xxs)
#     dbx = MeanShift().fit(Xxs)
    
    core_samples_mask = np.zeros_like(dbx.labels_, dtype=bool)
#    core_samples_mask[dbx.core_sample_indices_] = True
    labels = dbx.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print("n_cols total", n_clusters_)

    tricol = [labels[i:i+3] for i in range(0, len(labels), 3)] if triple else labels

    print([tcol for tcol in tricol])
    print(len(tricol), len(words))

    for word, col in zip(words, tricol): # majority vote if triple
        word['cluster-col'] = max(col, key=list(col).count) if triple else col
        
    # #############################################################################
    
    Xys = [t[1] for t in [Xys[i:i+3] for i in range(0, len(Xys), 3)]]
#     Xys = np.array(Xys).reshape(-1, 1)
    
    dby = DBSCAN(eps=0.02, min_samples=1).fit(Xys)
    core_samples_mask = np.zeros_like(dby.labels_, dtype=bool)
    core_samples_mask[dby.core_sample_indices_] = True
    labels = dby.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("n_rows total", n_clusters_)

    trirow = labels#[list(labels[i:i+3]) for i in range(0, len(labels), 3)]

    print(trirow)
    
    for word, row in zip(words, trirow):
        word['cluster-row'] = row#max(row, key=list(row).count) # majority vote
#     # #############################################################################
#     dby = DBSCAN(eps=0.05, min_samples=10).fit(Xys)
#     core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#     core_samples_mask[db.core_sample_indices_] = True
#     labels = db.labels_
#     # Number of clusters in labels, ignoring noise if present.
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise_ = list(labels).count(-1)

#     trirow = [list(labels[i:i+3]) for i in range(0, len(labels), 3)]

#     print(trirow)
    
#     for word, row in zip(words, trirow):
#         word['cluster-row'] = max(set(row), key=list(row).count) # majority vote
    
    return words