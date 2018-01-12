import os
import sys
import numpy as np
from skimage import io


def nomoarlize(face):
    face -= np.min(face)
    face /= np.max(face)
    face = (face * 255).astype(np.uint8)
    return face


filepath = sys.argv[1]
target = sys.argv[2]
target = os.path.join(filepath, target)
target_image = io.imread(target).reshape(600 * 600 * 3)

pics = os.listdir(filepath)
image = np.ndarray([len(pics), 600 * 600 * 3], dtype='float')

for i, img in enumerate(pics):
    photo = os.path.join(filepath, img)
    image[i] = io.imread(photo).reshape(600 * 600 * 3)


mean = image.mean(axis=0)
image_center = image - mean

U, S, V = np.linalg.svd(image_center.T, full_matrices=False)

X_center = target_image - image.mean(axis=0)
weights = np.dot(X_center, U)
recon = mean + np.dot(weights[:4], U[:, :4].T)

io.imsave('reconstruction.jpg', nomoarlize(recon).reshape(600, 600, 3))
