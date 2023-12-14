import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import *


sns.set(style='darkgrid')


def test_hipp():
    hm = HippocampusModel(4, 2)
    data_input = [
        [1., 1., .3, .6],
        [1., 1., .3, .6],
        [0., 0., .3, .6],
    ]
    data_output = [
        [1.],
        [1.],
        [0.],
    ]

    with tf.Session() as sess:
        # initialize
        sess.run(tf.global_variables_initializer())

        loss = hm.fit(data_input, data_output)
        print(loss)

        out = hm.predict(data_input)
        print(out)

def test_cort():
    hm = HippocampusModel(4, 2)
    cm = CorticalModel(hm, 4, 2)
    data_input = [
        [1., 1., .3, .6],
        [1., 1., .3, .6],
        [0., 0., .3, .6],
    ]
    data_output = [
        [1.],
        [1.],
        [0.],
    ]

    with tf.Session() as sess:
        # initialize
        sess.run(tf.global_variables_initializer())

        loss = cm.fit(data_input, data_output)
        print(loss)

        out = cm.predict(data_input)
        print(out)

def sensory_preconditioning():
    data_input1 = [
        [1., 1., .3, .6],
    ]
    data_output1 = [
        [0.],
    ]
    data_input2 = [
        [1., 0., .3, .6],
    ]
    data_output2 = [
        [1.],
    ]
    data_input3 = [
        [0., 1., .3, .6],
    ]

    def generate_normal_data():
        model = Model(4, 2, learning_rate=0.05)

        with tf.Session() as sess:
            # initialize
            sess.run(tf.global_variables_initializer())

            loss2, out2 = model.fit(data_input2, data_output2, n_hipp=500)

            final_out = model.predict(data_input3)

        return out2, final_out

    def generate_precondition_data():
        model = Model(4, 2, learning_rate=0.05)

        with tf.Session() as sess:
            # initialize
            sess.run(tf.global_variables_initializer())

            loss1, out1 = model.fit(data_input1, data_output1, n_hipp=500)
            loss2, out2 = model.fit(data_input2, data_output2, n_hipp=500)

            final_out = model.predict(data_input3)

        return (out1 + out2), final_out

    trace_normal, final_normal = generate_normal_data()
    trace_precond, final_precond = generate_precondition_data()
    plt.plot(np.linspace(500, 999, 500), np.squeeze(trace_normal), color='blue', label='Not Preconditioned')
    plt.plot(np.squeeze(trace_precond), color='orange', label='Preconditioned')
    plt.scatter(1000, final_normal, color='blue')
    plt.scatter(1000, final_precond, color='orange')
    plt.legend()
    plt.ylabel("Response")
    plt.xlabel("Iteration")
    plt.show()

def discrimination():
    data_input1 = [
        [1., 1., .3, .6],
    ]
    data_output1 = [
        [-1.],
    ]
    data_input2 = [
        [1., 0., .3, .6],
        [0., 1., .3, .6],
    ]
    data_output2 = [
        [1.],
        [-1.],
    ]

    def generate_normal_data():
        model = Model(4, 2)

        with tf.Session() as sess:
            # initialize
            sess.run(tf.global_variables_initializer())

            loss2, out2 = model.fit(data_input2, data_output2, n_hipp=100, n_cort=100)

        return out2

    def generate_precondition_data():
        model = Model(4, 2)

        with tf.Session() as sess:
            # initialize
            sess.run(tf.global_variables_initializer())

            loss1, out1 = model.fit(data_input1, data_output1, n_hipp=100, n_cort=100)
            loss2, out2 = model.fit(data_input2, data_output2, n_hipp=100, n_cort=100)

        return out1, out2

    def plot_discrim(offset, out, color, label):
        # print(out[0])
        tr0 = list(map(lambda e: e[0][0], out))
        tr1 = list(map(lambda e: e[1][0], out))
        plt.plot(np.linspace(offset, offset + len(tr0) - 1, len(tr0)), tr0, color=color, label=label)
        plt.plot(np.linspace(offset, offset + len(tr1) - 1, len(tr1)), tr1, color=color)


    trace_normal = generate_normal_data()
    trace_precond1, trace_precond2 = generate_precondition_data()

    plot_discrim(100, trace_normal, 'blue', 'Not Preconditioned')

    plt.plot(np.squeeze(trace_precond1), color='orange')
    plot_discrim(100, trace_precond2, 'orange', 'Preconditioned')
    plt.legend()
    plt.ylabel("Response")
    plt.xlabel("Iteration")
    plt.show()

def shock_deficit(lesioned=False):
    data_input1a = [
        [1., 0., .3, .6],
    ]
    data_input1b = [
        [1., 1., .3, .6],
    ]
    data_output1 = [
        [0.],
    ]

    data_input2 = [
        [1., 0., .3, .6],
    ]

    data_output2 = [
        [1.],
    ]

    data_input3 = [
        [0., 0., .3, .6],
    ]

    lr = 0.05

    def generate_normal_data():
        model = Model(4, 2, is_lesioned=lesioned, learning_rate=lr)

        with tf.Session() as sess:
            # initialize
            sess.run(tf.global_variables_initializer())

            loss1, out1 = model.fit(data_input1a, data_output1, n_hipp=100)
            loss2, out2 = model.fit(data_input2, data_output2, n_hipp=10, n_cort=100)
            finalout = model.predict(data_input3)
        return out1 + out2, finalout

    def generate_precondition_data():
        model = Model(4, 2, is_lesioned=lesioned, learning_rate=lr)

        with tf.Session() as sess:
            # initialize
            sess.run(tf.global_variables_initializer())

            loss1, out1 = model.fit(data_input1b, data_output1, n_hipp=100)
            # out1 = [0.] * 100
            loss2, out2 = model.fit(data_input2, data_output2, n_hipp=10, n_cort=100)
            finalout = model.predict(data_input3)

        return out1 + out2, finalout

    # def plot_discrim(offset, out, color, label):
    #     # print(out[0])
    #     tr0 = list(map(lambda e: e[0][0], out))
    #     tr1 = list(map(lambda e: e[1][0], out))
    #     plt.plot(np.linspace(offset, offset + len(tr0) - 1, len(tr0)), tr0, color=color, label=label)
    #     plt.plot(np.linspace(offset, offset + len(tr1) - 1, len(tr1)), tr1, color=color)

    trace_normal, final_normal = generate_normal_data()
    trace_precond, final_precond = generate_precondition_data()

    # plt.plot(np.squeeze(trace_precond), color='orange')
    # plot_discrim(100, trace_precond2, 'orange', 'Preconditioned')
    # plt.legend()
    # plt.ylabel("Response")
    # plt.xlabel("Iteration")
    # plt.show()


    plt.plot(np.squeeze(trace_normal), color='blue', label='Not Preconditioned')
    plt.plot(np.squeeze(trace_precond), color='orange', label='Preconditioned')
    plt.scatter(220, final_normal, color='blue')
    plt.scatter(220, final_precond, color='orange')
    plt.legend()
    plt.ylabel("Response")
    plt.xlabel("Iteration")
    plt.show()


# test_hipp()
# test_cort()
# sensory_preconditioning()
# discrimination()
shock_deficit()
