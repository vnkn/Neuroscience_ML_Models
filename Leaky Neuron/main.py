import random
import numpy as np
import matplotlib.pyplot as plt
from model import Model
import seaborn as sns

sns.set(style='darkgrid')


def run_trial(m, n_steps=1000, n_target=50, distractor_only=False):
    # outputs = np.zeros(n_steps)
    outputs = []
    for i in range(-100, n_steps - 100):
        # print(i)
        input_spike = 1.0 if (not distractor_only and i >= 0 and i < n_target) else 0.
        # input_spike = 0.
        distractor_spike = 1.0 if random.random() < .10 else 0.
        if m.step(input_spike, distractor_spike):
            outputs.append(i)
    return outputs

def generate_hist(m, ax, d_only=False):
    n_trials = 10
    counter = []
    for i in range(n_trials):
        counter += run_trial(m, distractor_only=d_only)
    print(counter)
    # counter /= n_trials
    # plt.hist(counter, bins=np.linspace(0.0, 1000.0, 50))
    sns.distplot(counter, bins=np.linspace(-100.0, 900.0, 50))
    plt.xlabel("Time (ms)")
    plt.ylabel("Spike Frequency")

def run_trial_switched(m, n_steps=1000, n_target=50):
    # outputs = np.zeros(n_steps)
    outputs = []
    for i in range(-100, n_steps - 100):
        # print(i)
        distractor_spike = 1.0 if (i >= 0 and i < n_target) else 0.
        # input_spike = 0.
        input_spike = 1.0 if random.random() < .10 else 0.
        if m.step(input_spike, distractor_spike):
            outputs.append(i)
    return outputs

def generate_hist_switched(m):
    n_trials = 10
    counter = []
    for i in range(n_trials):
        counter += run_trial_switched(m)
    print(counter)
    # counter /= n_trials
    # plt.hist(counter, bins=np.linspace(0.0, 1000.0, 50))
    sns.distplot(counter, bins=np.linspace(-100.0, 900.0, 50))
    plt.xlabel("Time (ms)")
    plt.ylabel("Spike Frequency")

def run_trial_lp(m, n_steps=1000, n_target=50):
    # outputs = np.zeros(n_steps)
    outputs = []
    for i in range(-100, n_steps - 100):
        # print(i)
        input_prob = .05 if (i >= 0 and i < n_target) else 0.
        input_spike = 1.0 if random.random() < input_prob else 0.
        # input_spike = 0.
        distractor_spike = 1.0 if random.random() < .10 else 0.
        if m.step(input_spike, distractor_spike):
            outputs.append(i)
    return outputs

def generate_hist_lp(m):
    n_trials = 10
    counter = []
    for i in range(n_trials):
        counter += run_trial_lp(m)
    print(counter)
    # counter /= n_trials
    # plt.hist(counter, bins=np.linspace(0.0, 1000.0, 50))
    sns.distplot(counter, bins=np.linspace(-100.0, 900.0, 50))
    plt.xlabel("Time (ms)")
    plt.ylabel("Spike Frequency")

def q1():
    # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    plt.subplot(121)
    m = Model(coupled=True)
    generate_hist(m, None)
    plt.title("Targets (coupling)")

    plt.subplot(122)
    mc = Model(coupled=False)
    generate_hist(mc, None)
    plt.title("Targets (no coupling)")
    plt.show()

    plt.subplot(121)
    m = Model(coupled=True)
    generate_hist(m, None, d_only=True)
    plt.title("Distractors (coupling)")

    plt.subplot(122)
    mc = Model(coupled=False)
    generate_hist(mc, None, d_only=True)
    plt.title("Distractors (no coupling)")
    plt.show()

def q2():
    # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    plt.subplot(121)
    m = Model(coupled=True)
    generate_hist_switched(m)
    plt.title("Switched Targets (coupling)")

    plt.subplot(122)
    mc = Model(coupled=False)
    generate_hist_switched(mc)
    plt.title("Switched Targets (no coupling)")
    plt.show()

def q3():
    plt.subplot(121)
    m = Model(coupled=True)
    generate_hist_lp(m)
    plt.title("Low Probability Targets (coupling)")

    plt.subplot(122)
    mc = Model(coupled=False)
    generate_hist_lp(mc)
    plt.title("Low Probability Targets (no coupling)")
    plt.show()


# q1()
# q2()
q3()
