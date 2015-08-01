import numpy as np
import problem_data
import utils
import utils_fns


def prob_3():
    m, n, k, T, p, A, B, c, P, q, r, pT, qT, rT, x0 = problem_data.hw6_p3_data()
    val_fns = []
    controllers = []
    next_cost = utils_fns.QuadraticFunction(pT, qT, rT)
    for t_val in range(T)[::-1]:
        total_cost = utils_fns.QuadraticFunction(np.zeros([n+m, n+m]), np.zeros([n+m, 1]), np.zeros(1))
        for k_val in range(k):
            A_val = A[t_val, k_val, :, :]
            B_val = B[t_val, k_val, :, :]
            c_val = c[t_val, k_val, :, :]
            P_val = P[t_val, k_val, :, :]
            q_val = q[t_val, k_val, :, :]
            r_val = r[t_val, k_val, 0]
            this_stage_quadratic = utils_fns.QuadraticFunction(P_val, q_val, r_val)
            linear_A = np.hstack((A_val, B_val))
            linear_to_next_stage = utils_fns.LinearFunction(linear_A, c_val)
            next_stage_quadratic = next_cost.precompose_linear(linear_to_next_stage)
            quadratic_this_k = this_stage_quadratic.plus_quadratic(next_stage_quadratic)
            total_cost = total_cost.plus_quadratic(quadratic_this_k)
        total_cost = total_cost.div_by_num(k)
        qfx, lfu = total_cost.partial_minimization(m)
        val_fns.append(qfx)
        controllers.append(lfu)
        next_cost = qfx
    utils.label('6.3')
    print 'Optimal expected total cost:', val_fns[-1].evaluate(x0)


if __name__ == '__main__':
    prob_3()
