import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image


def compress_image(image, num_cluster):
    h, w, d = image.shape
    new_input = image.reshape(h * w, d)
    km = KMeans(n_clusters=num_cluster).fit(new_input)
    labels = km.labels_.reshape(h, w)
    output = np.ones((h, w, d), dtype=int)
    for i in range(num_cluster):
        output[labels == i, :] = km.cluster_centers_[i].reshape(1, 1, 3)
    return output


# im = np.array(Image.open('house.jpg'))
# out = compress_image(im, 10)
# plt.imshow(out)
# plt.show()
