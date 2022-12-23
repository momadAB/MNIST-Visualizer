# Import relevant modules
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# Global variables
learning_rate = 0.003
epochs = 50
batch_size = 4000
validation_split = 0.2
dropout_rate = 0.6
if 0:
    show_plot = True
else:
    show_plot = False


# Returns normalized test and training data
def preprocess():
    # The following lines adjust the granularity of reporting.
    pd.options.display.max_rows = 10
    pd.options.display.float_format = "{:.1f}".format

    # The following line improves formatting when outputting NumPy arrays.
    np.set_printoptions(linewidth=200)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize the data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

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

    # The features are stored in a two-dimensional 28X28 array.
    # Flatten that two-dimensional array into a one-dimensional
    # 784-element array.
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    # Define the first hidden layer.
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))

    # Second hidden layer
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))

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

    history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                        epochs=epochs, shuffle=True,
                        validation_split=validation_split)

    # To track the progression of training, gather a snapshot
    # of the model's metrics at each epoch.
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist


# Run main function
def main():
    x_train, y_train, x_test, y_test = preprocess()
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
    my_model.evaluate(x=x_test, y=y_test, batch_size=batch_size)

    # Save model to disk
    my_model.save('my_model')

    print("Generate a prediction\n")
    plt.imshow(x_test[0])
    prediction = my_model.predict(x_test[:1])

    # Finds the number that the model predicts the input to be
    prediction_number = 0
    highest_value = 0
    n = 0

    for i in prediction.tolist()[0]:
        if i > highest_value:
            highest_value = i
            prediction_number = n
        n = n + 1

    print("prediction: ", prediction_number)


if __name__ == "__main__":
    main()
