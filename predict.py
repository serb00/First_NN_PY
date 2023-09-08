import random
import matplotlib.pyplot as plt
import utils
import numpy as np


def predict_random_image(filename):
    images, labels = utils.load_dataset()

    (weights_input_to_hidden, weights_hidden_to_output, bias_input_to_hidden,
     bias_hidden_to_output) = utils.load_variables(f"{filename}.npz")

    # Select random image from images
    rnd = random.randint(0, 60000)
    test_image = images[rnd]
    test_label = labels[rnd].argmax()

    # Predict
    image = np.reshape(test_image, (-1, 1))
    # Forward propogation (to hidden layer)
    hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
    hidden = 1 / (1 + np.exp(-hidden_raw))
    # Forward propogation (to output layer)
    output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
    output = 1 / (1 + np.exp(-output_raw))

    # Draw results
    plt.imshow(test_image.reshape(28, 28), cmap="Greys")
    plt.title(f"NN sugests the number is: {output.argmax()}"
              f",label is {test_label}")
    plt.show()


def find_error(filename):
    images, labels = utils.load_dataset()
    error_found = False
    correct_in_a_row = 0

    (weights_input_to_hidden, weights_hidden_to_output, bias_input_to_hidden,
     bias_hidden_to_output) = utils.load_variables(f"{filename}.npz")

    while not error_found:
        # Select random image from images
        rnd = random.randint(0, 60000)
        test_image = images[rnd]
        test_label = labels[rnd].argmax()
        correct_in_a_row += 1

        # Predict
        image = np.reshape(test_image, (-1, 1))
        # Forward propogation (to hidden layer)
        hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
        hidden = 1 / (1 + np.exp(-hidden_raw))
        # Forward propogation (to output layer)
        output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
        output = 1 / (1 + np.exp(-output_raw))

        error_found = output.argmax() != test_label

    # Draw results
    plt.imshow(test_image.reshape(28, 28), cmap="Greys")
    plt.title(f"NN sugests the number is: {output.argmax()}"
              f",label is {test_label}"
              f", error after {correct_in_a_row} attempts.")
    plt.show()
