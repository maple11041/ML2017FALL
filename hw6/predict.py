import numpy as np
import pandas as pd
from keras.models import model_from_json
from sklearn.cluster import KMeans
import sys

X_train = np.load(sys.argv[1])
X_train = X_train.astype('float32') / 255. - 0.5
X_train = ((X_train.T - np.mean(X_train, axis=1)) / (np.std(X_train, axis=1) + 1e-7)).T

json_file = open('encoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
encoder = model_from_json(loaded_model_json)
encoder.load_weights("encoder.h5")

encoded_imgs = encoder.predict(X_train, verbose=1)
kmeans = KMeans(n_clusters=2).fit(encoded_imgs)
y = kmeans.labels_

ans = np.zeros(1980000, dtype='int')
test = pd.read_csv(sys.argv[2])
for i in range(1980000):
    image1 = test['image1_index'][i]
    image2 = test['image2_index'][i]
    if y[image1] == y[image2]:
        ans[i] = 1

submit = pd.DataFrame({'Ans': ans})
submit.to_csv(sys.argv[3], index_label='ID')
