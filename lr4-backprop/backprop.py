import numpy as np

# Constants
INPUT_NEURONS = 4
HIDDEN_NEURONS = 3
OUTPUT_NEURONS = 4
LEARN_RATE = 0.2
MAX_SAMPLES = 18

# Weight Structures
wih = np.random.uniform(-0.5, 0.5, (INPUT_NEURONS + 1, HIDDEN_NEURONS))
who = np.random.uniform(-0.5, 0.5, (HIDDEN_NEURONS + 1, OUTPUT_NEURONS))

# Activations
inputs = np.zeros(INPUT_NEURONS)
hidden = np.zeros(HIDDEN_NEURONS)
target = np.zeros(OUTPUT_NEURONS)
actual = np.zeros(OUTPUT_NEURONS)

# Unit Errors
erro = np.zeros(OUTPUT_NEURONS)
errh = np.zeros(HIDDEN_NEURONS)

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
    [0.0, 1.0, 0.0, 1.0, [0.0, 0.0, 0.0, 1.0]]
]

strings = ["Attack", "Run", "Wander", "Hide"]

def sigmoid(val):
    return 1.0 / (1.0 + np.exp(-val))

def sigmoid_derivative(val):
    return val * (1.0 - val)

def feed_forward():
    global hidden, actual
    hidden = sigmoid(np.dot(inputs, wih[:-1]) + wih[-1])
    actual = sigmoid(np.dot(hidden, who[:-1]) + who[-1])

def back_propagate():
    global wih, who
    erro = (target - actual) * sigmoid_derivative(actual)
    errh = np.dot(erro, who[:-1].T) * sigmoid_derivative(hidden)
    who[:-1] += LEARN_RATE * np.outer(hidden, erro)
    who[-1] += LEARN_RATE * erro
    wih[:-1] += LEARN_RATE * np.outer(inputs, errh)
    wih[-1] += LEARN_RATE * errh

def action(vector):
    return np.argmax(vector)

def main():
    global inputs, target
    iterations = 0
    sum_correct = 0

    while iterations <= 100000:
        sample = samples[iterations % MAX_SAMPLES]
        inputs = np.array(sample[:INPUT_NEURONS])
        target = np.array(sample[INPUT_NEURONS])
        feed_forward()
        err = 0.5 * np.sum((target - actual) ** 2)
        # print(f"mse = {err}")
        back_propagate()
        iterations += 1

    sum_correct = 0
    for sample in samples:
        inputs = np.array(sample[:INPUT_NEURONS])
        target = np.array(sample[INPUT_NEURONS])
        feed_forward()
        if action(actual) == action(target):
            sum_correct += 1
        else:
            print(f"{inputs} {strings[action(actual)]} ({strings[action(target)]})")

    print(f"Network is {sum_correct / MAX_SAMPLES * 100.0}% correct")

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