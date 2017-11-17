from keras.models import model_from_json
import sys
import hw3_input
import pandas as pd


def main():
    json_file = open('model/adamax250.json', 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights('model/adamax250.h5')
    X_test = hw3_input.load_test_data(sys.argv[1])
    result = model.predict(X_test, batch_size=128, verbose=1)
    Y_test = result.argmax(axis=1)
    submission = pd.DataFrame({'label': Y_test})
    submission.to_csv(sys.argv[2], index_label='id', index=True)


if __name__ == '__main__':
    main()
