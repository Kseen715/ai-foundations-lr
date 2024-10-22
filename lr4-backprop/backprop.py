import numpy as np

# Constants
INPUT_NEURONS = 4
HIDDEN_NEURONS = 3
OUTPUT_NEURONS = 4
LEARN_RATE = 0.2
# MAX_SAMPLES = 18

# Weight Structures
weights_input_hidden = np.random.uniform(-0.5, 0.5, (INPUT_NEURONS + 1, HIDDEN_NEURONS))
weights_hidden_output = np.random.uniform(-0.5, 0.5, (HIDDEN_NEURONS + 1, OUTPUT_NEURONS))

# Activations
inputs = np.zeros(INPUT_NEURONS)
hidden = np.zeros(HIDDEN_NEURONS)
target = np.zeros(OUTPUT_NEURONS)
actual = np.zeros(OUTPUT_NEURONS)

# Samples
samples = [
    [2.0, 0.0, 0.0, 0.0, [0.0, 0.0, 1.0, 0.0]],
    [2.0, 0.0, 0.0, 1.0, [0.0, 0.0, 1.0, 0.0]],
    [2.0, 0.0, 1.0, 1.0, [1.0, 0.0, 0.0, 0.0]],
    [2.0, 0.0, 1.0, 2.0, [1.0, 0.0, 0.0, 0.0]],
    [2.0, 1.0, 0.0, 2.0, [0.0, 0.0, 0.0, 1.0]],
    [2.0, 1.0, 0.0, 1.0, [1.0, 0.0, 0.0, 0.0]],
    [1.0, 0.0, 0.0, 0.0, [0.0, 0.0, 1.0, 0.0]],
    [1.0, 0.0, 0.0, 1.0, [0.0, 0.0, 0.0, 1.0]],
    [1.0, 0.0, 1.0, 1.0, [1.0, 0.0, 0.0, 0.0]],
    [1.0, 0.0, 1.0, 2.0, [0.0, 0.0, 0.0, 1.0]],
    [1.0, 1.0, 0.0, 2.0, [0.0, 0.0, 0.0, 1.0]],
    [1.0, 1.0, 0.0, 1.0, [0.0, 0.0, 0.0, 1.0]],
    [0.0, 0.0, 0.0, 0.0, [0.0, 0.0, 1.0, 0.0]],
    [0.0, 0.0, 0.0, 1.0, [0.0, 0.0, 0.0, 1.0]],
    [0.0, 0.0, 1.0, 1.0, [0.0, 0.0, 0.0, 1.0]],
    [0.0, 0.0, 1.0, 2.0, [0.0, 1.0, 0.0, 0.0]],
    [0.0, 1.0, 0.0, 2.0, [0.0, 1.0, 0.0, 0.0]],
    [0.0, 1.0, 0.0, 1.0, [0.0, 0.0, 0.0, 1.0]],
    [2.0, 0.0, 0.0, 0.0, [0.0, 0.0, 1.0, 0.0]],
    [2.0, 0.0, 0.0, 1.0, [0.0, 0.0, 1.0, 0.0]],
    [2.0, 0.0, 1.0, 1.0, [1.0, 0.0, 0.0, 0.0]],
    [2.0, 0.0, 1.0, 2.0, [1.0, 0.0, 0.0, 0.0]],
    [2.0, 1.0, 0.0, 2.0, [0.0, 0.0, 0.0, 1.0]],
    [2.0, 1.0, 0.0, 1.0, [1.0, 0.0, 0.0, 0.0]],
    [1.0, 0.0, 0.0, 0.0, [0.0, 0.0, 1.0, 0.0]],
    [1.0, 0.0, 0.0, 1.0, [0.0, 0.0, 0.0, 1.0]],
    [1.0, 0.0, 1.0, 1.0, [1.0, 0.0, 0.0, 0.0]],
    [1.0, 0.0, 1.0, 2.0, [0.0, 0.0, 0.0, 1.0]],
    [1.0, 1.0, 0.0, 2.0, [0.0, 0.0, 0.0, 1.0]],
    [1.0, 1.0, 0.0, 1.0, [0.0, 0.0, 0.0, 1.0]],
    [0.0, 0.0, 0.0, 0.0, [0.0, 0.0, 1.0, 0.0]],
    [0.0, 0.0, 0.0, 1.0, [0.0, 0.0, 0.0, 1.0]],
    [0.0, 0.0, 1.0, 1.0, [0.0, 0.0, 0.0, 1.0]],
    [0.0, 0.0, 1.0, 2.0, [0.0, 1.0, 0.0, 0.0]],
    [0.0, 1.0, 0.0, 2.0, [0.0, 1.0, 0.0, 0.0]],
    [0.0, 1.0, 0.0, 1.0, [0.0, 0.0, 0.0, 1.0]]
]

strings = ["Attack", "Run", "Wander", "Hide"]

def sigmoid(val):
    return 1.0 / (1.0 + np.exp(-val))

def sigmoid_derivative(val):
    return val * (1.0 - val)

def feed_forward():
    global hidden, actual
    hidden = sigmoid(np.dot(inputs, weights_input_hidden[:-1]) + weights_input_hidden[-1])
    actual = sigmoid(np.dot(hidden, weights_hidden_output[:-1]) + weights_hidden_output[-1])

def back_propagate():
    global weights_input_hidden, weights_hidden_output
    erro = (target - actual) * sigmoid_derivative(actual)
    errh = np.dot(erro, weights_hidden_output[:-1].T) * sigmoid_derivative(hidden)
    # np.outer is used to calculate the outer product of two vectors
    weights_hidden_output[:-1] += LEARN_RATE * np.outer(hidden, erro)
    weights_hidden_output[-1] += LEARN_RATE * erro
    weights_input_hidden[:-1] += LEARN_RATE * np.outer(inputs, errh)
    weights_input_hidden[-1] += LEARN_RATE * errh

def action(vector):
    return np.argmax(vector)

def save_model(filemane="weights.npz"):
    global weights_input_hidden, weights_hidden_output, INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS
    # zip the weights and save them to a file
    with open(filemane, "wb") as f:
        np.savez_compressed(f, INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS, weights_input_hidden, weights_hidden_output)

def load_model(filemane="weights.npz"):
    # load the weights from a file
    global weights_input_hidden, weights_hidden_output, INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS
    with open(filemane, "rb") as f:
        data = np.load(f)
        INPUT_NEURONS = data["arr_0"]
        HIDDEN_NEURONS = data["arr_1"]
        OUTPUT_NEURONS = data["arr_2"]
        weights_input_hidden = data["arr_3"]
        weights_hidden_output = data["arr_4"]

def train():
    global inputs, target
    iterations = 0
    sum_correct = 0

    # shuffle the samples
    np.random.shuffle(samples)

    # 70:30 split
    train, test = samples[:len(samples) * 7 // 10], samples[len(samples) * 7 // 10:] 

    while iterations <= 100000:
        print('', end="\r")
        sample = train[iterations % len(train)]
        inputs = np.array(sample[:INPUT_NEURONS])
        target = np.array(sample[INPUT_NEURONS])
        feed_forward()
        err = 0.5 * np.sum((target - actual) ** 2)
        print(f"mse = {err}", end="")
        back_propagate()
        iterations += 1
    print()

    sum_correct = 0
    for sample in test:
        inputs = np.array(sample[:INPUT_NEURONS])
        target = np.array(sample[INPUT_NEURONS])
        feed_forward()
        if action(actual) == action(target):
            sum_correct += 1
        else:
            print(f"{inputs} {strings[action(actual)]} ({strings[action(target)]})")

    print(f"Network is {sum_correct / len(test) * 100.0}% correct")

    save_model()

def main():
    global inputs, target, weights_hidden_output, weights_input_hidden
    train()

    load_model()

    test_cases = [
        [2.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0],
        [2.0, 0.0, 1.0, 3.0],
        [2.0, 1.0, 0.0, 3.0],
        [0.0, 1.0, 0.0, 3.0]
    ]

    for test in test_cases:
        inputs = np.array(test)
        feed_forward()
        print(f"{test} Action {strings[action(actual)]}")

if __name__ == "__main__":
    main()