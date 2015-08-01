import problem_data
import numpy as np
import utils_io


###########
# problem 1
###########
def prob_1():
    W, s, t = problem_data.hw4_p1_data()
    utils_io.label('4.1b')
    print 'Shortest paths for each matrix in \'matrix_name, [path], weight\' format:'
    for i in range(W.shape[0]):
        p, wp = bellman_ford(W[i], s[i], t[i])
        print 'matrix_{0:s}, {1:s}, {2:s}'.format(str(i), str(p), str(wp))


def test_bellman_ford():
    W = np.array([[np.inf, 5, 3, np.inf, np.inf, 5, np.inf],
                  [np.inf, np.inf, 10, 18, np.inf, np.inf, np.inf],
                  [np.inf, np.inf, np.inf, 12, 10, np.inf, 1],
                  [np.inf, np.inf, np.inf, np.inf, 6, np.inf, np.inf],
                  [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
                  [np.inf, np.inf, 4, np.inf, np.inf, np.inf, 7],
                  [np.inf, np.inf, np.inf, np.inf, 8, np.inf, np.inf]])
    start_node = 0
    end_node = 4
    p, wp = bellman_ford(W, start_node, end_node)
    print 'shortest path from ' + str(start_node) + ' to ' + str(end_node) + ':', p
    print 'shortest path length from ' + str(start_node) + ' to ' + str(end_node) + ':', str(wp)


def bellman_ford(W, v_start, v_end):
    n_vert = W.shape[0]
    v_last = np.inf * np.ones(W.shape[0])
    v_last[v_end] = 0
    paths = [[]] * n_vert
    while True:
        v_last_mat = np.repeat(v_last, n_vert).reshape([n_vert, n_vert]).T
        g_val = np.min(W + v_last_mat, axis=1)
        to_append = np.argmin(W + v_last_mat, axis=1)
        v_next = np.where(v_last < g_val, v_last, g_val)
        for idx, vert in enumerate(range(n_vert)):
            if g_val[idx] < v_last[idx]:
                paths[idx] = paths[to_append[idx]] + [to_append[idx]]
        if np.array_equal(v_last, v_next):
            break
        v_last = v_next
    forward_paths = [path[::-1] for path in paths]
    return forward_paths[v_start], v_last[v_start]


###########
# problem 2
###########
def prob_2():
    W, s, t = problem_data.hw4_p1_data()
    utils_io.label('4.2c')
    print 'Shortest paths for each matrix in \'matrix_name, [path], weight\' format:'
    for i in range(W.shape[0]):
        p, wp = forward_bellman_ford(W[i], s[i], t[i])
        print 'matrix_{0:s}, {1:s}, {2:s}'.format(str(i), str(p), str(wp))


def test_forward_bellman_ford():
    W = np.array([[np.inf, 5, 3, np.inf, np.inf, 5, np.inf],
                  [np.inf, np.inf, 10, 18, np.inf, np.inf, np.inf],
                  [np.inf, np.inf, np.inf, 12, 10, np.inf, 1],
                  [np.inf, np.inf, np.inf, np.inf, 6, np.inf, np.inf],
                  [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
                  [np.inf, np.inf, 4, np.inf, np.inf, np.inf, 7],
                  [np.inf, np.inf, np.inf, np.inf, 8, np.inf, np.inf]])
    start_node = 0
    end_node = 4
    p, wp = forward_bellman_ford(W, start_node, end_node)
    print 'shortest path from ' + str(start_node) + ' to ' + str(end_node) + ':', p
    print 'shortest path length from ' + str(start_node) + ' to ' + str(end_node) + ':', str(wp)


def forward_bellman_ford(W, v_start, v_end):
    n_vert = W.shape[0]
    v_last = np.inf * np.ones(W.shape[0])
    v_last[v_start] = 0
    paths = [[]] * n_vert
    while True:
        v_last_mat = np.repeat(v_last, n_vert).reshape([n_vert, n_vert]).T
        g_val = np.min(W.T + v_last_mat, axis=1)
        to_append = np.argmin(W.T + v_last_mat, axis=1)
        v_next = np.where(v_last < g_val, v_last, g_val)
        for idx, vert in enumerate(range(n_vert)):
            if g_val[idx] < v_last[idx]:
                paths[idx] = paths[to_append[idx]] + [to_append[idx]]
        if np.array_equal(v_last, v_next):
            break
        v_last = v_next
    return paths[v_end], v_last[v_end]


###########
# problem 3
###########
def prob_3():
    a, b, weights, n = problem_data.hw4_p3_data()
    num_nodes = n + 2
    b = b[np.argsort(a)]
    b = np.concatenate((np.array([0]), b, np.array([np.inf])))
    weights = weights[np.argsort(a)]
    weights = np.concatenate((np.array([0]), weights, np.array([0])))
    a = a[np.argsort(a)]
    a = np.concatenate((np.array([0]), a, np.array([np.inf])))
    W = np.inf * np.ones([num_nodes, num_nodes])
    for i in range(W.shape[0]):
        for j in range(W.shape[0]):
            if a[j] >= b[i] and j > i:
                W[i, j] = -weights[i]
            elif a[j] >= a[i] and j > i:
                W[i, j] = 0
    p, wp = bellman_ford(W, 0, num_nodes - 1)
    utils_io.label('4.3b')
    print 'Maximum weight job schedule (assuming jobs labeled 1 to n): ', p[:-1]
    print 'Total weight: ', -wp


if __name__ == '__main__':
    # test_bellman_ford()
    prob_1()
    # test_forward_bellman_ford()
    prob_2()
    prob_3()
