from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys
import hw3_input


batch_size = 128
nb_classes = 7
nb_epoch = 250
img_rows, img_cols = 48, 48
img_channels = 1


def build_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(48, 48, 1), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.summary()
    return model


def main():
    X_train, y_train = hw3_input.load_train_data(sys.argv[1])
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    model = build_model()
    model.summary()
    datagen = ImageDataGenerator(
        shear_range=0.2,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        # randomly rotate images in the range (degrees, 0 to 180)
        rotation_range=10,
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.15,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.15,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)
    datagen.fit(X_train)
    checkpointer = ModelCheckpoint(filepath='./checkpoint/check.hdf5', verbose=1, save_best_only=True, monitor='acc')
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        steps_per_epoch=int(np.ceil(X_train.shape[0] / float(batch_size))),
                        epochs=nb_epoch, callbacks=[checkpointer])
    model_json = model.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model/model.h5")


if __name__ == '__main__':
    main()
