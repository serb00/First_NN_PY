import numpy as np


def load_dataset():
    with np.load("mnist.npz") as f:
        # convert from RGB to Unit RGB
        x_train = f['x_train'].astype("float32") / 255

        # reshape from (60000, 28, 28) to (60000, 784)
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
        # x_train.reshape((60000, 28*28))

        # labels
        y_train = f['y_train']

        # convert to output layer format
        y_train = np.eye(10)[y_train]

        return x_train, y_train


def save_variables(filename, w_i_h, w_h_o, b_i_h, b_h_o):
    np.savez(filename,
             weights_input_to_hidden=w_i_h,
             weights_hidden_to_output=w_h_o,
             bias_input_to_hidden=b_i_h,
             bias_hidden_to_output=b_h_o)


def load_variables(filename):
    data = np.load(filename)
    return (data['weights_input_to_hidden'],
            data['weights_hidden_to_output'],
            data['bias_input_to_hidden'],
            data['bias_hidden_to_output'])
