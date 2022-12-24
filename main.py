# Import relevant modules
import keras.models
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from gridmaker import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Global variables
learning_rate = 0.001
epochs = 5
batch_size = 96
validation_split = 0.2
dropout_rate = 0.5
if 0:  # Show plot
    show_plot = True
else:
    show_plot = False
if 0:  # Train or load
    training = True
else:
    training = False
if 1:  # Make grid or not
    make_grid_bool = True
else:
    make_grid_bool = False


# Returns normalized test and training data
def preprocess():
    # The following lines adjust the granularity of reporting.
    pd.options.display.max_rows = 10
    pd.set_option('display.precision', 2)

    # The following line improves formatting when outputting NumPy arrays.
    np.set_printoptions(linewidth=300)

    # datagen = ImageDataGenerator(
    #     rotation_range=20,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     zoom_range=[0.4, 0],
    #     horizontal_flip=False,
    #     fill_mode='nearest')

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)  # IDK why but needed
    x_test = np.expand_dims(x_train, axis=-1)  # IDK why but needed

    # Set all values above 0 to 1
    x_train[x_train > 125] = 255
    x_test[x_test > 125] = 255

    # Normalize the data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # datagen.fit(x_train)
    # plt.imshow(x_train[2])
    # print(f"{y_train[2]}")

    """ 
    x_train contains the training set's features.
    y_train contains the training set's labels.
    x_test contains the test set's features.
    y_test contains the test set's labels.
    """
    return x_train, y_train, x_test, y_test


# Define the plotting function
def plot_curve(epoch, hist, list_of_metrics):
    """Plot a curve of one or more classification metrics vs. epoch."""
    # list_of_metrics should be one of the names shown in:
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epoch[1:], x[1:], label=m)

    plt.legend()
    if show_plot:
        plt.show()


def create_model(my_learning_rate):
    """Create and compile a deep neural net."""

    # All models in this course are sequential.
    model = tf.keras.models.Sequential()

    # Convolution layers
    model.add(tf.keras.layers.Conv2D(32, 3, activation='relu',  padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(2))

    model.add(tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(2))

    # The features are stored in a two-dimensional 28X28 array.
    # Flatten that two-dimensional array into a one-dimensional
    # 784-element array.
    model.add(tf.keras.layers.Flatten())

    # Define the first hidden layer.
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))

    # Second hidden layer
    # model.add(tf.keras.layers.Dense(units=128, activation='relu'))

    # Define a dropout regularization layer.
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))

    # Define the output layer. The units parameter is set to 10 because
    # the model must choose among 10 possible output values (representing
    # the digits from 0 to 9, inclusive).
    #
    # Don't change this layer.
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    # Construct the layers into a model that TensorFlow can execute.
    # Notice that the loss function for multi-class classification
    # is different from the loss function for binary classification.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    return model


def train_model(model, train_features, train_label, epochs,
                batch_size=None, validation_split=0.1):
    """Train the model by feeding it data."""

    # datagen = ImageDataGenerator(
    #     rotation_range=20,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     zoom_range=[0.4, 0],
    #     horizontal_flip=False,
    #     fill_mode='nearest')

    datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=[0.2, 0])

    # history = model.fit(train_features, train_label, batch_size=batch_size,
    #                     epochs=epochs, shuffle=True,
    #                     validation_split=validation_split)

    datagen.fit(train_features)
    history = model.fit(datagen.flow(train_features, train_label, batch_size=batch_size), batch_size=batch_size,
                        epochs=epochs, shuffle=True,
                        validation_data=datagen.flow(train_features, train_label, batch_size=32))

    # To track the progression of training, gather a snapshot
    # of the model's metrics at each epoch.
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist


# Inputs Tkinter drawing and returns the models prediction
def predict_model(model, drawing_input):
    """Predict the output of the model by feeding it data."""
    prediction = model.predict(drawing_input)
    prediction_number = 0
    highest_value = 0
    n = 0
    # Finds the predicted number by the model
    for i in prediction.tolist()[0]:
        if i > highest_value:
            highest_value = i
            prediction_number = n
        n = n + 1

    return prediction_number


# Run main function
def main():
    x_train, y_train, x_test, y_test = preprocess()
    if training:  # Train model
        # Establish the model's topography.
        my_model = create_model(learning_rate)

        # Train the model on the normalized training set.
        epoch, hist = train_model(my_model, x_train, y_train,
                                  epochs, batch_size, validation_split)

        # Plot a graph of the metric vs. epochs.
        list_of_metrics_to_plot = ['accuracy']
        plot_curve(epoch, hist, list_of_metrics_to_plot)

        # Evaluate against the test set.
        print("\n Evaluate the new model against the test set:")
        # my_model.evaluate(x=x_test, y=y_test, batch_size=batch_size)

        # Save model to disk
        my_model.save('my_model')
    else:  # Load the model
        my_model = keras.models.load_model('my_model')

    if make_grid_bool:
        # Make grid/canvas
        make_grid(my_model)

    # print("Generate a prediction\n")
    # plt.imshow(x_test[0])
    # prediction = my_model.predict(x_test[:1])
    # print(x_test[:1])

    # Finds the number that the model predicts the input to be
    # prediction_number = 0
    # highest_value = 0
    # n = 0
    #
    # for i in prediction.tolist()[0]:
    #     if i > highest_value:
    #         highest_value = i
    #         prediction_number = n
    #     n = n + 1
    #
    # print("prediction:", prediction_number)


if __name__ == "__main__":
    main()
