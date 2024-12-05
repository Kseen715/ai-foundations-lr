import zipfile
import json
import io
import math
from pprint import pprint

from PIL import Image
import numpy as np
import colorama as clr

# # Constants
# input_neurons = None
# hidden_neurons = None
# output_neurons = None

# # Weight Structures
# weights_input_hidden = None
# weights_hidden_output = None

# # Activations
# inputs = None
# hidden = None
# target = None
# actual = None

def sigmoid(val):
    return 1.0 / (1.0 + np.exp(-val))


def sigmoid_derivative(val):
    return val * (1.0 - val)


def feed_forward():
    global hidden, actual
    hidden = sigmoid(
        np.dot(inputs, weights_input_hidden[:-1]) + weights_input_hidden[-1])
    actual = sigmoid(
        np.dot(hidden, weights_hidden_output[:-1]) + weights_hidden_output[-1])


def back_propagate():
    global weights_input_hidden, weights_hidden_output
    erro = (target - actual) * sigmoid_derivative(actual)
    errh = np.dot(
        erro, weights_hidden_output[:-1].T) * sigmoid_derivative(hidden)
    # np.outer is used to calculate the outer product of two vectors
    weights_hidden_output[:-1] += LEARN_RATE * np.outer(hidden, erro)
    weights_hidden_output[-1] += LEARN_RATE * erro
    weights_input_hidden[:-1] += LEARN_RATE * np.outer(inputs, errh)
    weights_input_hidden[-1] += LEARN_RATE * errh


def action(vector):
    return np.argmax(vector)


def save_model(filemane="weights.npz"):
    global weights_input_hidden, weights_hidden_output, input_neurons, hidden_neurons, output_neurons
    # zip the weights and save them to a file
    with open(filemane, "wb") as f:
        np.savez_compressed(f, input_neurons, hidden_neurons,
                            output_neurons, weights_input_hidden, weights_hidden_output)


def load_model(filemane="weights.npz"):
    # load the weights from a file
    global weights_input_hidden, weights_hidden_output, input_neurons, hidden_neurons, output_neurons
    with open(filemane, "rb") as f:
        data = np.load(f)
        input_neurons = data["arr_0"]
        hidden_neurons = data["arr_1"]
        output_neurons = data["arr_2"]
        weights_input_hidden = data["arr_3"]
        weights_hidden_output = data["arr_4"]


def train(its=100000):
    global inputs, target, samples, input_neurons
    iterations = 0
    its_digits = len(str(its))
    sum_correct = 0

    # shuffle the samples
    np.random.shuffle(samples)

    # 70:30 split
    train, test = samples[:len(samples) * 7 //
                          10], samples[len(samples) * 7 // 10:]

    print_freq = 20
    try:
        last_errs = []
        while iterations <= its:
            if iterations % print_freq == 0:
                print('', end="\r")
            sample = train[iterations % len(train)]
            inputs = np.array(sample[:input_neurons])
            target = np.array(sample[input_neurons])
            feed_forward()
            err = 0.5 * np.sum((target - actual) ** 2)
            last_errs.append(err)
            if len(last_errs) > 1000:
                last_errs.pop(0)
            # find the mean squared error
            merr = np.mean(last_errs)
            if iterations % print_freq == 0:
                print(f"i: {iterations:0{its_digits}d}, mean: {merr:0.3f}, mse: {err}", end="")
            back_propagate()
            iterations += 1
        print()
    except KeyboardInterrupt:
        print()
        print(f"Training stopped at iteration {iterations}")
        print(f"Mean squared error: {merr}")
        print(f"Error: {err}")

    print(f'{clr.Fore.BLUE}TEST ON TRAIN DATA:{clr.Style.RESET_ALL}')

    errors = {}
    sum_correct = 0
    for sample in train:
        inputs = np.array(sample[:input_neurons])
        target = np.array(sample[input_neurons])
        feed_forward()
        if action(actual) == action(target):
            sum_correct += 1
        else:
            key = f"{strings[action(actual)]} ({strings[action(target)]})"
            if key not in errors:
                errors[key] = 0
            errors[key] += 1
            # print()

    print(f"{clr.Fore.RED}Errors: {clr.Style.RESET_ALL}")
    pprint(errors)
    print(f"Network is {sum_correct / len(train) * 100.0}% correct")

    print(f'{clr.Fore.BLUE}TEST ON TEST DATA:{clr.Style.RESET_ALL}')

    errors = {}
    sum_correct = 0
    for sample in test:
        inputs = np.array(sample[:input_neurons])
        target = np.array(sample[input_neurons])
        feed_forward()
        if action(actual) == action(target):
            sum_correct += 1
        else:
            key = f"{strings[action(actual)]} ({strings[action(target)]})"
            if key not in errors:
                errors[key] = 0
            errors[key] += 1
            # print()

    print(f"{clr.Fore.RED}Errors: {clr.Style.RESET_ALL}")
    pprint(errors)
    print(f"Network is {sum_correct / len(test) * 100.0}% correct")

    save_model()


def read_zip_to_data_array(filename):
    global samples, strings
    classes_count = len(strings)
    res = []

    with zipfile.ZipFile(filename, "r") as zi:
        in_filelist = zi.filelist
        for i in in_filelist:
            file = i.filename
            item_type = file.split("_")[0]
            # print(file, item_type)

            byte_data = zi.read(i)
            img = Image.open(io.BytesIO(byte_data))

            pixels = img.getdata()
            pixel_bytes = img.tobytes()
            len_pixels = len(pixels)
            # print(f'Needed input neurons: {len_pixels}')
            # print(pixels, pixel_bytes)

            # map bytes to 0-1
            fpixels = [x / 255.0 for x in pixel_bytes]
            # load
            # print(fpixels)

            needed_neurons = math.ceil(math.log(classes_count, 2))
            # assert needed_neurons == output_neurons, f'Needed output neurons: req({needed_neurons}) != prov({output_neurons})'
            # print(f'Needed output neurons: {needed_neurons}')

            # map to binary
            binary = [int(x) for x in bin(int(item_type))[2:]]
            # fill with zeros
            binary = [0] * (needed_neurons - len(binary)) + binary
            # print(binary)

            fpixels.append(binary)

            res.append(fpixels)

    samples = res
    return samples
    # pprint(samples)


def read_json_names(filename):
    global strings
    with open(filename, "r") as f:
        res = json.loads(f.read())
        pprint(res)
        # index is in the value
        strings = [k for k, v in res.items()]
        # print(strings)


def main():
    global inputs, target, weights_hidden_output, weights_input_hidden, input_neurons, hidden_neurons, output_neurons, samples, strings, LEARN_RATE, actual, hidden

    # Constants
    hidden_neurons = 768
    output_neurons = 3
    LEARN_RATE = 0.00125

    sides = 32
    count = 10000
    input_neurons = sides * sides
    print(f"Input neurons: {input_neurons}")
    print(f"Hidden neurons: {hidden_neurons}")
    print(f"Output neurons: {output_neurons}")
    print(f"Learn rate: {LEARN_RATE}")
    read_json_names(f"data/names_{sides}x{sides}_{count}.json")
    read_zip_to_data_array(f"data/processed_{sides}x{sides}_{count}.zip")

    # Weight Structures
    weights_input_hidden = np.random.uniform(-0.5,
                                             0.5, (input_neurons + 1, hidden_neurons))
    weights_hidden_output = np.random.uniform(
        -0.5, 0.5, (hidden_neurons + 1, output_neurons))

    # Activations
    inputs = np.zeros(input_neurons)
    hidden = np.zeros(hidden_neurons)
    target = np.zeros(output_neurons)
    actual = np.zeros(output_neurons)

    train(100000)

    load_model()

    # test_cases = [
    #     [2.0, 1.0, 1.0, 1.0],
    #     [1.0, 1.0, 1.0, 2.0],
    #     [0.0, 0.0, 0.0, 0.0],
    #     [0.0, 1.0, 1.0, 1.0],
    #     [2.0, 0.0, 1.0, 3.0],
    #     [2.0, 1.0, 0.0, 3.0],
    #     [0.0, 1.0, 0.0, 3.0]
    # ]

    # for test in test_cases:
    #     inputs = np.array(test)
    #     feed_forward()
    #     print(f"{test} Action {strings[action(actual)]}")


if __name__ == "__main__":
    main()
