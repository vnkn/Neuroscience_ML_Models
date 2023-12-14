import numpy as np
from tqdm import tqdm
from model import Model
from image import load_inputs_from_file, weights_to_ascii
from random import shuffle

import matplotlib.pyplot as plt


ITERS = 500


def visualize_weights(m):
    weights = m.get_weights()
    for n, w in enumerate(weights):
        print("{}\n{}".format(n, weights_to_ascii(w)))


def evaluate_model(m, inputs, verbose=False):
    variances = []
    for j, frames in enumerate(inputs):
        if verbose:
            print("Sequence ", j)
        outputs = []
        for frame in frames:
            outputs.append(m.run(frame))
        if verbose:
            print(outputs)
        variances.append(np.var(outputs))
    return np.sum(variances)

def main():

    inputs = load_inputs_from_file("data1.txt")

    m = Model()
    var_trace = []
    y_bars = []

    # Train model on input sequences
    for i in tqdm(range(ITERS)):
        for j, frames in enumerate(inputs):
            shuffle(frames)
            for frame in frames:
                m.train(frame)
                y_bars.append(m.get_y_bars())
        m.train(np.zeros((8,8,4)))
        var_trace.append(evaluate_model(m, inputs))
        if i % 100 == 0 or i == 10:
            visualize_weights(m)
            # evaluate_model(m, inputs, verbose=True)

    # Visualize weights
    print("FINAL WEIGHTS:")
    visualize_weights(m)


    print("EVALUATION:")
    for j, frames in enumerate(inputs):
        print("Sequence ", j)
        for frame in frames:
            print(m.run(frame))

    plt.plot(var_trace)
    plt.xlabel("Iteration")
    plt.ylabel("Total Variance")
    plt.show()

    for j in range(4):
        plt.plot([yb[j] for yb in y_bars], label='$y_{}$'.format(j))
    plt.xlabel("Iteration")
    plt.ylabel("Trace Value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
