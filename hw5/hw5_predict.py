import numpy as np
import pandas as pd
import sys
from keras.models import model_from_json


json_file = open('modeldim50.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("modeldim50.h5")

test = pd.read_csv(sys.argv[1])
std = np.load('std.npy')
mean = np.load('mean.npy')
user_test = test['UserID'].values - 1
movie_test = test['MovieID'].values - 1

result = model.predict([user_test, movie_test], verbose=1)
result = np.squeeze(result)
result = result * std + mean
result = np.clip(result, 1, 5)


submission = pd.DataFrame({'Rating': result})
submission.index += 1
submission.to_csv(sys.argv[2], index_label='TestDataID')
