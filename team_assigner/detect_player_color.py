import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
def detect_color(crop_image):
    half_crop_image = crop_image[0:int(crop_image.shape[0]/2), :]
    image_2d = half_crop_image.reshape(-1, 3)
    kmean = KMeans(n_clusters=2, random_state=0)
    kmean.fit(image_2d)
    label = kmean.labels_
    clustered_image = label.reshape(half_crop_image.shape[0], half_crop_image.shape[1])
    corner_cluster = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
    background_label = max(set(corner_cluster), key=corner_cluster.count)
    player_label = 1 - background_label
    return kmean.cluster_centers_[player_label]