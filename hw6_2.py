import numpy as np
import utils_io


def exact_stage_cost(A, B, Q, R, W, n):
    P_arr = []
    K_arr = []

    T = 100

    dummy_terminal_val = np.zeros([n, n])
    P_arr.append(dummy_terminal_val)
    K_arr.append(dummy_terminal_val)

    for _ in range(T)[::-1]:
        P_next = P_arr[-1]
        term_a = -np.linalg.inv((R + np.dot(np.dot(B.T, P_next), B)))
        term_b = np.dot(np.dot(B.T, P_next), A)
        K = np.dot(term_a, term_b)
        K_arr.append(K)

        k_r_k = np.dot(np.dot(K.T, R), K)
        a_bk = A + np.dot(B, K)
        P = Q + k_r_k + np.dot(np.dot(a_bk.T, P_next), a_bk)

        P_arr.append(P)

    P_ss = P_arr[-1]
    K_ss = K_arr[-1]
    utils_io.label('6.2')
    print 'optimal policy:', K_ss
    print 'calculated avg stage cost:', 0.5 * np.trace(np.dot(P_ss, W))

    return K_ss


def simulated_stage_cost(A, B, K_ss, Q, R, W, n):
    T = 10000
    costs = []
    x = np.zeros(n)
    W_mean = np.zeros(n)

    for time in range(T):
        u = np.dot(K_ss, x)
        cost = 0.5 * (np.dot(np.dot(x.T, Q), x) + np.dot(np.dot(u.T, R), u))
        costs.append(cost)
        w_sample = np.random.multivariate_normal(W_mean, W)
        x = np.dot(A, x) + np.dot(B, u) + w_sample
    print 'simulated average stage cost: ', np.mean(costs[int(T/10):])


def prob_2():
    n = 4
    rho = 0.1
    sigma = 1

    A = np.eye(n)
    B = np.array([[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1], [0, 0, 0, 1]])
    W = np.zeros([n, n])
    W[n-1, n-1] = sigma ** 2
    Q = np.eye(n)
    R = rho * np.eye(n)

    K_ss = exact_stage_cost(A, B, Q, R, W, n)
    simulated_stage_cost(A, B, K_ss, Q, R, W, n)


if __name__ == '__main__':
    prob_2()