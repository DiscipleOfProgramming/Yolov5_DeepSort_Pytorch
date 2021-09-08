from math import ceil
import os
import numpy
import numpy as np
from sklearn.cluster import KMeans
import cv2 as cv
import sys

video_path = r'D:\Final_dataset\including_overlap\data\event2105.mp4'
features_path = r'feature_vectors.txt'

def cluster_nparr():
    with open(features_path, 'r') as file:
        text = file.read()
        bboxes = text.split("?;")
        num_boxes = len(bboxes) - 1
        row_list = []
        all_features = None
        # might need to normalize the data
        for i in range(num_boxes):
            elements = bboxes[i].split(',')
            frame_idx, id, x1, y1, x2, y2, features = elements
            features_list = features.strip("[").strip("]").split()
            if i < num_boxes / 2:
                row_list.append(features_list)
                i = 0
            elif i == ceil(num_boxes / 2):
                all_features = np.array(row_list)
                print(all_features)
                kmeans = KMeans(n_clusters=3).fit(all_features)
            else:
                feature_vec = np.array(features_list)
                pred = kmeans.predict(feature_vec)
                i = 9

def cluster_():
    with open(features_path, 'r') as file:
        text = file.read()
        bboxes = text.split("?;")
        num_boxes = len(bboxes) - 1
        all_features = np.zeros((ceil(num_boxes / 2), 512))
        # might need to normalize the data
        for i in range(num_boxes):
            elements = bboxes[i].split(',')
            frame_idx, id, x1, y1, x2, y2, features = elements
            features_list = features.strip("[").strip("]").split()
            feature_vec = np.array(features_list)
            if i < num_boxes / 2:
                all_features[i,:] = feature_vec
                i = 0
            elif i == ceil(num_boxes / 2):
                print(all_features)
                kmeans = KMeans(n_clusters=3, random_state=0).fit(all_features)
            else:
                pred = kmeans.predict([feature_vec])
                i = 9


        kmeans = KMeans(n_clusters=3, random_state=0).fit(all_features)
        pass

if __name__ == "__main__":
    # cluster_nparr()
    cluster_()