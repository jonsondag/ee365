import numpy as np

######################
# utils for MDPs (hw5)
# grouped here since they are used across hw files
######################
def _add_g_to_next_values(g, v_next, f_w):
    """
    :param g: stage cost for state x and input u; g.shape = [n, m]
    :param v_next: next time period costs for state x; v_next.shape = [n]
    :param f_w: value at n, m is next state idx given curr state n, input m; f_w.shape = [n, m]

    :return: current state value [n, m]
    """
    v_next_for_eval = v_next[:, np.newaxis]  # make v_next 2-D to allow advanced indexing
    v_next_x_idx = f_w.ravel().astype(int)
    v_next_y_idx = np.zeros(len(v_next_x_idx)).astype(int)
    v_next = v_next_for_eval[v_next_x_idx, v_next_y_idx]
    v_next = v_next.reshape(g.shape)
    return g + v_next


def value(f, g, g_final, w_dist, T, g_is_w_dependent=False, g_is_time_dependent=False):
    """
    inputs
    :param f: x_t1 = f(x_t, u_t, w_t); f.shape = [n, m, p]
    :param g: stage cost for state x and input u;
        g.shape is [n, m] if g is not dependent on w or time
        g.shape is [n, m, p] if g is dependent on w only
        g.shape is [n, m, T] if g is dependent on time only
        g.shape is [n, m, p, T] if g is dependent on w and time
    :param g_final: final stage cost; g_final.shape = [n]
    :param w_dist: distribution of random disturbance that happens after input choice;
        w_dist.shape = [p]
    :param T: number of time periods
    :param g_is_w_dependent: True/False depending on whether g has w dependence
    :param g_is_time_dependent: True/False depending on whether g varies with time

    :return: pol: policy for state x and time t; pol.shape = [n, t]
    :return: v: value at state x and time t; val.shape = [n, t]

    n: number of states
    m: number of inputs
    p: number of random disturbance values
    """
    n = f.shape[0]
    m = f.shape[1]
    p = f.shape[2]

    pol = np.zeros([n, T])  # n by T
    v = np.zeros([n, T+1])  # n by (T + 1)
    v[:, T] = g_final
    for time in range(0, T)[::-1]:
        e_v_over_w = np.zeros([n, m])
        for w_idx in range(p):
            f_w = f[:, :, w_idx]
            v_next = v[:, time + 1]
            if g_is_w_dependent and not g_is_time_dependent:
                values_this_w = _add_g_to_next_values(g[:, :, w_idx], v_next, f_w)
            elif g_is_time_dependent and not g_is_w_dependent:
                values_this_w = _add_g_to_next_values(g[:, :, time], v_next, f_w)
            elif g_is_time_dependent and g_is_w_dependent:
                values_this_w = _add_g_to_next_values(g[:, :, w_idx, time], v_next, f_w)
            else:
                values_this_w = _add_g_to_next_values(g, v_next, f_w)
            e_v_over_w += values_this_w * w_dist[w_idx]
        v[:, time] = np.min(e_v_over_w, axis=1)
        pol[:, time] = np.argmin(e_v_over_w, axis=1)
    return pol.astype(int), v


def cloop(f, g, pol, w_dist=None, g_is_w_dependent=False, g_is_time_dependent=False):
    """
    :param f: x_t1 = f(x_t, u_t, w_t); f.shape = [n, m, p]
    :param g: stage cost for state x and input u;
        g.shape is [n, m] if g is not dependent on w or time
        g.shape is [n, m, p] if g is dependent on w only
        g.shape is [n, m, T] if g is dependent on time only
        g.shape is [n, m, p, T] if g is dependent on w and time
    :return: pol: policy for state x and time t; pol.shape = [n, T]
    :param g_is_time_dependent: True/False depending on whether g varies with time
    :return: fcl: closed-loop dynamics; fcl.shape = [n, p, T]
    :return: gcl: closed-loop cost; gcl.shape = [n, T]
    """
    if w_dist is None:
        if g_is_w_dependent:
            raise Exception('Need to provide w_dist if g_is_w_dependent')
        else:
            w_dist = np.ones(1)

    n = f.shape[0]
    p = f.shape[2]
    T = pol.shape[1]

    fcl = np.zeros([n, p, T])
    gcl = np.zeros([n, T])
    f_zero_index = np.arange(n).repeat(p).reshape(n, p)  # manually coerce array to n by p for advanced indexing
    f_two_index = np.tile(np.arange(p), n).reshape(n, p)  # manually coerce array to n by p for advanced indexing
    g_zero_index = np.arange(n)
    for time in range(T):
        f_one_index = pol[:, time].repeat(p).reshape(n, p)  # manually coerce array to n by p for advanced indexing
        fcl[:, :, time] = f[f_zero_index, f_one_index, f_two_index]
        g_one_index = pol[:, time]
        for w_idx, w in enumerate(w_dist):
            if g_is_w_dependent and not g_is_time_dependent:
                gcl[:, time] += w * g[g_zero_index, g_one_index, w_idx]
            elif g_is_time_dependent and not g_is_w_dependent:
                gcl[:, time] += w * g[g_zero_index, g_one_index, time]
            elif g_is_time_dependent and g_is_w_dependent:
                gcl[:, time] += w * g[g_zero_index, g_one_index, w_idx, time]
            else:
                gcl[:, time] += w * g[g_zero_index, g_one_index]
    fcl = fcl.astype(int)
    return fcl, gcl


def ftop(fcl, w_dist):
    """
    :param: fcl: closed-loop dynamics; fcl.shape = [n, p, T]
    :param w_dist: distribution of random disturbance that happens after input choice;
        w_dist.shape = [p]
    :return P: time-varying transition matrix. P.shape = [n, n, T]
    """
    n = fcl.shape[0]
    T = fcl.shape[2]
    P = np.zeros([n, n, T])
    for time in range(T):
        P_time = np.zeros([n, n])
        for w_idx, w in enumerate(w_dist):
            P_time[np.arange(n), fcl[:, w_idx, time]] += w
        P[:, :, time] = P_time
    return P

######################
# utils for 'other information pattern' MDPs (hw5),
# where we learn (random) information about the current state
# before making a decision
# grouped here since they are used across hw files
######################
def _add_g_to_next_values_info_pat(g, v_next, f_w):
    """
    :param g: stage cost for state x and input u; g.shape = [n, m, p1]
    :param v_next: next time period costs for state x; v_next.shape = [n]
    :param f_w: value at n, m is next state idx given curr state n, input m; f_w.shape = [n, m, p1]

    :return: current state value [n, m, p1]
    """
    v_next_for_eval = v_next[:, np.newaxis, np.newaxis]  # make v_next 3-D to allow advanced indexing
    v_next_x_idx = f_w.ravel().astype(int)
    v_next_y_idx = np.zeros(len(v_next_x_idx)).astype(int)
    v_next_z_idx = np.zeros(len(v_next_x_idx)).astype(int)
    v_next = v_next_for_eval[v_next_x_idx, v_next_y_idx, v_next_z_idx]
    v_next = v_next.reshape(g.shape)
    return g + v_next


def value_info_pat(f, g, g_final, w1_dist, w2_dist, T, g_is_time_dependent=False):
    """
    inputs
    :param f: x_t1 = f(x_t, u_t, w1_t, w2_t); f.shape = [n, m, p1, p2]
    :param g: stage cost for state x and input u;
        g.shape is [n, m, p1, p2] if g is dependent on w only
        g.shape is [n, m, p1, p2, T] if g is dependent on w and T
    :param g_final: final stage cost; g_final.shape = [n]
    :param w1_dist: distribution of random disturbance that happens before input choice;
        w1_dist.shape = [p1]
    :param w2_dist: distribution of random disturbance that happens after input choice;
        w2_dist.shape = [p2]
    :param T: number of time periods
    :param g_is_time_dependent: True/False depending on whether g varies with time

    :return: pol: policy for state x, w1_dist val w1, and time t; pol.shape = [n, p1, t]
    :return: v: value at state x and time t; val.shape = [n, t]

    n: number of states
    m: number of inputs
    p1: number of states, first random disturbance values (known at decision time)
    p2: number of states, second random disturbance values (not known at decision time; similar to w_dist in value() fn
    """
    n = f.shape[0]
    m = f.shape[1]
    p1 = f.shape[2]
    p2 = f.shape[3]

    pol = np.zeros([n, p1, T])
    v = np.zeros([n, T+1])
    v[:, T] = g_final
    for time in range(0, T)[::-1]:
        e_v_over_w2 = np.zeros([n, m, p1])
        for w2_idx in range(p2):
            f_w = f[:, :, :, w2_idx]
            v_next = v[:, time + 1]
            if g_is_time_dependent:
                values_this_w = _add_g_to_next_values_info_pat(g[:, :, :, w2_idx, time], v_next, f_w)
            else:
                values_this_w = _add_g_to_next_values_info_pat(g[:, :, :, w2_idx], v_next, f_w)
            e_v_over_w2 += values_this_w * w2_dist[w2_idx]
        pol[:, :, time] = np.argmin(e_v_over_w2, axis=1)
        v_min = np.min(e_v_over_w2, axis=1)
        v[:, time] = np.dot(v_min, w1_dist)
    return pol.astype(int), v


def cloop_info_pat(f, g, pol, w2_dist, g_is_time_dependent=False):
    """
    :param f: [n, m, p1, p2]; f.shape = [n, m, p1, p2]
    :param g: stage cost for state x and input u;
        g.shape = [n, m, p1, p2] if g is dependent on w only
        g.shape = [n, m, p1, p2, T] if g is dependent on w and T
    :param pol: policy for state x, w1_dist val w1, and time t; pol.shape = [n, p1, t]
    :param g_is_time_dependent: True/False depending on whether g varies with time
    :return: fcl: closed-loop dynamics; fcl.shape = [n, p1, p2, T]
    :return: gcl: closed-loop cost; gcl.shape = [n, p1, T]
    """
    n = f.shape[0]
    p1 = f.shape[2]
    p2 = f.shape[3]
    T = pol.shape[2]

    fcl = np.zeros([n, p1, p2, T])
    gcl = np.zeros([n, p1, T])

    f_zero_index = np.arange(n)[:, np.newaxis, np.newaxis]
    f_two_index = np.arange(p1)[np.newaxis, :, np.newaxis]
    f_three_index = np.arange(p2)[np.newaxis, np.newaxis, :]
    g_zero_index = np.arange(n)[:, np.newaxis]
    g_two_index = np.arange(p1)[np.newaxis, :]

    for time in range(T):
        f_one_index = pol[:, :, time].repeat(p2).reshape([n, p1, p2])
        fcl[:, :, :, time] = f[f_zero_index, f_one_index, f_two_index, f_three_index]
        g_one_index = pol[:, :, time].reshape([n, p1])
        for w2_idx, w2 in enumerate(w2_dist):
            gcl[:, :, time] += w2 * g[g_zero_index, g_one_index, g_two_index, w2_idx]

    fcl = fcl.astype(int)
    return fcl, gcl


def ftop_info_pat(fcl, w1_dist, w2_dist):
    """
    :param: fcl: closed-loop dynamics; fcl.shape = [n, p1, p2, T]
    :param w1_dist: distribution of random disturbance that happens before input choice;
        w1_dist.shape = [p1]
    :param w2_dist: distribution of random disturbance that happens after input choice;
        w2_dist.shape = [p2]
    :return P: time-varying transition matrix. P.shape = [n, n, T]
    """
    n = fcl.shape[0]
    T = fcl.shape[3]
    P = np.zeros([n, n, T])
    for time in range(T):
        P_time = np.zeros([n, n])
        for w1_idx, w1 in enumerate(w1_dist):
            for w2_idx, w2 in enumerate(w2_dist):
                P_time[np.arange(n), fcl[:, w1_idx, w2_idx, time]] += w1 * w2
        P[:, :, time] = P_time
    return P
