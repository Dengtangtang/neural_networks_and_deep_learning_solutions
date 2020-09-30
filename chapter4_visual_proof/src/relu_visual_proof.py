''' Proof on "ReLU has universality for computation."

    Idea
        one neuron with relu as activation function is not possible to approximate a step function,
        what about two such neurons?
'''


import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return max(0, x)


def step(x, w, h, s):
    ''' Difference of two parallel relus with a large enough weight
        will make a step function.

        Parameters
        ----------
        x : float
            x coordinate.
        w : float
            weight of the two hidden neurons, and also the slope of the relu functions.
        h : float|int
            height of the step function.
        s : float
            step size of the first relu function.

        return
        ------
        out : float
            corresponding y coordinate.
    '''

    b = -w * s  # bias of first hidden neuron
    s_ = h / w  # gap between first and second relu functions
    b_ = -w * (s_+s)  # bias of second hidden neuron

    return relu(w*x+b) - relu(w*x+b_)


if __name__ == '__main__':
    X = np.arange(-4.0, 8.0, 0.05)
    w = 50
    h = 3
    s = 2
    b = -w * s
    Y = np.array([step(x, w, h, s) for x in X])
    plt.plot(X, Y)
    plt.show()
