import numpy as np
import utils


def get_K(A_nom, B, P, R):
    term_1 = -np.linalg.inv(R + np.dot(np.dot(B.T, P), B))
    term_2 = np.dot(np.dot(B.T, P), A_nom)
    return np.dot(term_1, term_2)


def get_P(A_nom, B, K, P, Q, R, n, epsilon):
    term_1 = Q + np.dot(np.dot(K.T, R), K)
    a_bk = A_nom + np.dot(B, K)
    term_2 = np.dot(np.dot(a_bk.T, P), a_bk)
    term_3 = (epsilon ** 2 / 3) * np.trace(P) * np.eye(n)  # additional P term to account for noise in A
    return term_1 + term_2 + term_3


def riccati_recursion(A_nom, B, Q, R, recursion_steps, n, m, epsilon):
    K_arr = [np.zeros([n, m])]
    P_arr = [np.zeros([n, n])]

    for time in range(recursion_steps):
        K = get_K(A_nom, B, P_arr[-1], R)
        P = get_P(A_nom, B, K, P_arr[-1], Q, R, n, epsilon)
        K_arr.append(K)
        P_arr.append(P)
        # print 'k_diff:', str(np.linalg.norm(K_arr[-1] - K_arr[-2], ord=None))
        # print 'k_mean:', K_arr[-1].mean()

    return K_arr[-1], P_arr[-1]


def prob_1():
    A_nom = np.array([[0.9, -0.2, 0.4], [-0.1, -0.9, 0.3], [-0.8, -0.1, 1.2]])
    B = np.array([0.1, -0.1, 0.2])[:, np.newaxis]
    n = 3
    m = 1
    Q = np.eye(n)
    R = np.array([m])[:, np.newaxis]
    W = np.eye(n)
    epsilon = 0.3
    recursion_steps = 50

    K, P = riccati_recursion(A_nom, B, Q, R, recursion_steps, n, m, epsilon)

    utils.label('6.1b')
    print 'K*:', K
    print 'J*, optimal average stage cost:', np.trace(np.dot(P, W))


if __name__ == '__main__':
    prob_1()
