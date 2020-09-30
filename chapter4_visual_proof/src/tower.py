''' Visual proof that single-hidden-layer neural network will fit all functions.

    Problem Link
        http://neuralnetworksanddeeplearning.com/chap4.html#problem_863961

    Description
        "We've seen how to use networks with TWO hidden layers to approximate an arbitrary function.
        Can you find a proof showing that it's possible with just a SINGLE hidden layer?"
'''


import numpy as np
import matplotlib.pyplot as plt


def plot(X, Y, Z):
    ''' Plot 3-d graph.

        Parameters
        ----------
        X : numpy.ndarray
            x coordinates
        Y : numpy.ndarray
            y coordinates
        Z : numpy.ndarray
            z coordinates

        Return
        ------
        out : None
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def step(w_x, X, Y, x_0, y_0, alpha):
    ''' A step function in 3-d, the network behind it is two neurons in input layer and
        one neuron in hidden layer.

        The weight for w_y and b is calculated by:
            y - y_0 = tan(alpha) * (x - x_0) ... (1)
            w_x*x + w_y*y + b = 0 ... (2)

        Parameters
        ----------
        w_x : float
            weight for x
        X : numpy.ndarray
            x coordinates
        Y : numpy.ndarray
            y coordinates
        x_0 : float|int
            line on x-y plane passing through point (x_0, y_0)
        y_0 : float|int
            line on x-y plane passing through point (x_0, y_0)
        alpha : float
            line on x-y plane intersects x-axis at angle alpha,
            which means the value of slope of the line is `tan(alpha)`.

            The range of alpha is restricted to [0, 2*pi). For convinience,
            any alpha larger than `pi` will be subtracted from `pi`.

        Return
        ------
        out : numpy.ndarray
            sigmoid(weighted output from hidden neuron plus b)
            = σ(w_x*x + w_y*y + b)

            The range of output is (0, 1)
    '''

    if alpha >= 2 * np.pi or alpha < 0:
        raise Exception('Invalid alpha.')

    if alpha >= np.pi:
        alpha -= np.pi

    if alpha == 0 or alpha == np.pi:
        # When step function is along x-axis.
        w_y = w_x
        w_x = 0
        b = -w_y * y_0
    elif alpha == np.pi / 2 or alpha == 3 * np.pi / 2:
        # When step function is along y-axis.
        w_y = 0
        b = -w_x * x_0
    else:
        w_y = - w_x / np.tan(alpha)
        b = w_y * (x_0*np.tan(alpha) - y_0)

    B = np.full(X.shape, b)
    Z = sigmoid(w_x * X + w_y * Y + B)

    return Z


def bump(w_x, X, Y, x_0, y_0, d, alpha):
    ''' Merge two step functions, the parallel lines on x-y plane are centered at (x_0, y_0)
        with distance `d` to each line.

        Parameters
        ----------
        w_x : float
            weight for x
        X : numpy.ndarray
            x coordinates
        Y : numpy.ndarray
            y coordinates
        x_0 : float|int
            bump centered at (x_0, y_0)
        y_0 : float|int
            bump centered at (x_0, y_0)
        d : float
            the distance is restricted in (0, 1]
        alpha: float
            parallel lines on x-y plane intersect x-axis at angle alpha,
            which means the value of slope of the line is `tan(alpha)`.

            The range of alpha is restricted to [0, 2*pi).

        Return
        ------
        out : numpy.ndarray
            step2 - step1, a combination of two parallel step functions
            with the middle gap `2d`.

            The range of output is (0, 1)
    '''

    if not 0 < d <= 1:
        raise Exception('Invalid distance `d`.')

    if alpha == 0 or alpha == np.pi:
        # When bump function is along x-axis.
        y_1 = y_0 + d
        y_2 = y_0 - d
        x_1 = x_0
        x_2 = x_0
    elif alpha == np.pi / 2 or alpha == 3 * np.pi / 2:
        # When bump function is along y-axis.
        x_1 = x_0 + d
        x_2 = x_0 - d
        y_1 = y_0
        y_2 = y_0
    else:
        # Other directions.
        x_d = d * np.sqrt(np.power(np.tan(alpha), 2) / (1 + np.power(np.tan(alpha), 2)))
        y_d = d * np.sqrt(1 / (1 + np.power(np.tan(alpha), 2)))

        if np.tan(alpha) > 0:
            x_1 = x_0 + x_d
            y_1 = y_0 - y_d
            x_2 = x_0 - x_d
            y_2 = y_0 + y_d
        else:
            x_1 = x_0 + x_d
            y_1 = y_0 + y_d
            x_2 = x_0 - x_d
            y_2 = y_0 - y_d

    Z_1 = step(w_x, X, Y, x_1, y_1, alpha)
    Z_2 = step(w_x, X, Y, x_2, y_2, alpha)

    return Z_2 - Z_1


def tower(w_x, X, Y, x_0, y_0, d, alphas, m):
    ''' Merge two step functions, the parallel lines on x-y plane are centered at (x_0, y_0)
        with distance `d` to each line.

        The network behind the tower function is composed of input layer with two neurons and
        2m neurons(2 neurons for a bump) in a hidden layer, since I do not control the height
        of the tower. If I need to control the height just return the value with `h * (Z / m)`
        and `h` will be the weights related to this tower on the output layer.

        Parameters
        ----------
        m : int
            m bumps in m directions
        Other parameters are the same as `bump` function.

        Return
        ------
        out : numpy.ndarray
            m bumps' combination and divided by m. Hence, the non-overlapping part will
            be suppressed to 1/m. When m is large, 1/m -> zero. On the other hand, the
            overlapping part (tower) will be m/m = 1.
    '''

    if m <= 0 or not isinstance(m, int):
        raise Exception('m shall be a positive integer.')

    Z = np.zeros(X.shape)
    for alpha in alphas:
        Z += bump(w_x, X, Y, x_0, y_0, d, alpha)

    return Z / m


def sigmoid(z):
    # return 1 / (1 + np.exp(-z))  # could overflow
    return 0.5 * (1 + np.tanh(0.5 * z))


if __name__ == '__main__':
    x = np.arange(-8.0, 8.0, 0.05)
    y = np.arange(-8.0, 8.0, 0.05)
    X, Y = np.meshgrid(x, y, indexing='xy')  # coordinate mesh

    # --- step ---
    # w_x = 100
    # x_0 = 0
    # y_0 = 0
    # alpha = 0.75*np.pi
    # Z = step(w_x, X, Y, x_0, y_0, alpha)
    # plot(X, Y, Z)

    # --- bump ---
    # w_x = 100
    # x_0 = 0
    # y_0 = 0
    # d = 1
    # alpha = 0.5 * np.pi
    # Z = bump(w_x, X, Y, x_0, y_0, d, alpha)
    # plot(X, Y, Z)

    # --- tower ---
    # m = 30
    # w_x = 100
    # x_0 = 5
    # y_0 = 5
    # d = 1
    # alphas = [i*2*np.pi/m for i in range(m)]
    # Z = tower(w_x, X, Y, x_0, y_0, d, alphas, m)
    # plot(X, Y, Z)

    # --- two towers ---
    # m = 30
    # w_x = 100
    # x_1 = 5
    # y_1 = 5
    # x_2 = -5
    # y_2 = -5
    # d = 1
    # alphas = [i*2*np.pi/m for i in range(m)]
    # Z_1 = tower(w_x, X, Y, x_1, y_1, d, alphas, m)
    # Z_2 = tower(w_x, X, Y, x_2, y_2, d, alphas, m)
    # Z = Z_1 + Z_2
    # plot(X, Y, Z)
