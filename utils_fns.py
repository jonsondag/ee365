import numpy as np


def to_quad_mat(P, q, r=None):
    n = P.shape[0] + 1
    m = P.shape[1] + 1
    quad_mat = np.zeros([n, m])
    quad_mat[:n-1, :m-1] = P
    quad_mat[:n-1, m-1] = q.ravel()
    if r is not None:
        quad_mat[n-1, :m-1] = q.ravel()
        quad_mat[n-1, m-1] = r
    else:
        quad_mat[n-1, m-1] = 1
    return quad_mat


def from_quad_mat(Q):
    n = Q.shape[0]
    m = Q.shape[1]
    P = Q[:n-1, :m-1]
    if m != n:
        q = Q[:n-1, m-1]
    else:
        q1 = Q[:n-1, m-1]
        q2 = Q[n-1, :m-1]
        q = (q1 + q2) / 2
    q.shape = (n-1, 1)
    r = Q[n-1, m-1]
    return P, q, r


class LinearFunction(object):
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def __repr__(self):
        return 'A:' + str(self.A) + '\n' +\
               'b:' + str(self.b)

    def plus_linear(self, other):
        return LinearFunction(self.A + other.A, self.b + other.b)

    def times_constant(self, constant):
        return LinearFunction(constant * self.A, constant * self.b)

    def evaluate(self, x):
        return np.dot(self.A, x) + self.b

    def compose(self, other):
        A_result = np.dot(self.A, other.A)
        b_result = self.b + np.dot(self.A, other.b)
        return LinearFunction(A_result, b_result)

    def norm_squared_linear(self):
        """
        compute the norm-squared of a linear function
        :return: a quadratic function
        """
        A_b = np.hstack((self.A, self.b))
        dp = np.dot(A_b.T, A_b)
        n = dp.shape[0]
        P = 2 * dp[:n-1, :n-1]
        q = 2 * dp[n-1, :n-1]
        r = 2 * dp[n-1, n-1]
        return QuadraticFunction(P, q, r)


class QuadraticFunction(object):
    def __init__(self, P, q, r):
        self.P = P
        self.q = q
        self.r = r

    def __repr__(self):
        return 'P:' + str(self.P) + '\n' +\
        'q:' + str(self.q) + '\n' +\
        'r: ' + str(self.r)

    def plus_quadratic(self, other):
        return QuadraticFunction(self.P + other.P, self.q + other.q, self.r + other.r)

    def times_constant(self, constant):
        return QuadraticFunction(constant * self.P, constant * self.q, constant * self.r)

    def evaluate(self, x):
        return (0.5 * np.dot(np.dot(x.T, self.P), x) + np.dot(self.q.T, x) + 0.5 * self.r)[0][0]

    def precompose_linear(self, linear_function):
        quad_mat_this = to_quad_mat(self.P, self.q, self.r)
        quad_mat_lin = to_quad_mat(linear_function.A, linear_function.b)
        new_mat = np.dot(np.dot(quad_mat_lin.T, quad_mat_this), quad_mat_lin)
        P, q, r = from_quad_mat(new_mat)
        return QuadraticFunction(P, q, r)

    def partial_minimization(self, m):
        """
        Compute the partial minimization of the quadratic function qf over the last m entries
        :return qfx, the minimum value as a quadratic function
        :return lfu, the minimizer as a linear function
        """
        P_xx, P_ux, P_uu, q_x, q_u, r = self._get_partial_min_components(m)
        P_xu = P_ux.T

        P_qfx = P_xx - np.dot(np.dot(P_xu, np.linalg.inv(P_uu)), P_ux)
        q_qfx = q_x - np.dot(np.dot(P_xu, np.linalg.inv(P_uu)), q_u)
        r_qfx = r - np.dot(np.dot(q_u.T, np.linalg.inv(P_uu)), q_u)
        qfx = QuadraticFunction(P_qfx, q_qfx, r_qfx)

        A = np.dot(-np.linalg.inv(P_uu), P_ux)
        b = np.dot(-np.linalg.inv(P_uu), q_u)
        lfu = LinearFunction(A, b)
        return qfx, lfu

    def partial_minimization_with_constraint(self, m, constraints_A, constraints_b):
        """
        Compute the partial minimization of the quadratic function qf over the last m entries, subject to
          constraints laid out in constraints_A and constraints_b by solving the system of KKT equations (note,
          pipes are used for block matrix notation here, not to indicate absolute values).
          See e.g. Boyd & Vanderberghe, Example 5.1 for more information.

          | P_uu   A.T | * | u                    | = | -P_ux | * | x | + | -q_n |
          | A      0   |   | lagrange_multipliers |   | 0     |           | b    |

        constraints_A and constraints_b specify that:

          | constraints_A | * | u | = | constraints_b |

        :return qfx, the minimum value as a quadratic function
        :return lfu, the minimizer as a linear function
        """
        P_xx, P_ux, P_uu, q_x, q_u, r = self._get_partial_min_components(m)

        num_constraints = constraints_A.shape[0]
        num_x_vars = P_xx.shape[0]
        num_u_vars = P_uu.shape[0]

        lh_upper = np.hstack([P_uu, constraints_A.T])
        lh_lower = np.hstack([constraints_A, np.zeros([num_constraints, num_constraints])])
        lh = np.vstack([lh_upper, lh_lower])
        lh_inv = np.linalg.inv(lh)
        lh_inv_u = lh_inv[:num_u_vars, :]

        linfunc_linear_upper = np.eye(num_x_vars)
        linfunc_linear_lower = np.dot(lh_inv_u, np.vstack([-P_ux, np.zeros([num_constraints, num_x_vars])]))
        linfunc_linear = np.vstack([linfunc_linear_upper, linfunc_linear_lower])
        linfunc_constant_upper = np.zeros([num_x_vars, 1])
        linfunc_constant_lower = np.dot(lh_inv_u, np.vstack([-q_u, constraints_b]))
        linfunc_constant = np.vstack([linfunc_constant_upper, linfunc_constant_lower])
        linfunc = LinearFunction(linfunc_linear, linfunc_constant)

        qfx = self.precompose_linear(linfunc)
        lfu = LinearFunction(linfunc_linear_lower, linfunc_constant_lower)

        return qfx, lfu

    def partial_evaluation(self, y, m):
        """
        compute the partial evaluation of this quadratic function with the last m entries of the
        argument equal to y.
        :return qf2, the value of this quadratic as a function of the remaining variables
        """
        P_xx, P_yx, P_yy, q_x, q_y, r = self._get_partial_min_components(m)
        P_new = P_xx
        q_new = (np.dot(y.T, P_yx) + q_x.T).T
        r_new = r + np.dot(np.dot(y.T, P_yy), y) + 2 * np.dot(q_y.T, y)
        return QuadraticFunction(P_new, q_new, r_new)

    def partial_expectation(self, y_bar, y_var):
        """
        compute the partial expectation of this quadratic function with respect to the last
          length(y_bar) entries, when those entries have mean y_bar and variance y_var
        :return: a new quadratic function
        """
        m = len(y_bar)
        _, _, P_yy, _, _, _ = self._get_partial_min_components(m)
        other_r = np.trace(np.dot(P_yy, y_var))
        partial_quad_func = self.partial_evaluation(y_bar, m)
        other_quad_func = QuadraticFunction(np.zeros(partial_quad_func.P.shape), np.zeros(partial_quad_func.q.shape),
                                            other_r)
        return partial_quad_func.plus_quadratic(other_quad_func)

    def _get_partial_min_components(self, m):
        P_m = self.P.shape[0]
        P_xx = self.P[0:P_m - m, 0:P_m - m]
        P_ux = self.P[P_m - m:, 0:P_m - m]
        P_uu = self.P[P_m - m:, P_m - m:]
        q_x = self.q[0:P_m - m]
        q_u = self.q[P_m - m:]
        r = self.r

        return P_xx, P_ux, P_uu, q_x, q_u, r

    def div_by_num(self, k):
        return QuadraticFunction(self.P / k, self.q / k, self.r / k)

    def difference(self, other, include_r):
        self_coefs = np.concatenate((self.P.ravel(), self.q.ravel()))
        other_coefs = np.concatenate((other.P.ravel(), other.q.ravel()))
        if include_r:
            self_coefs = np.concatenate((self_coefs.ravel(), self.r.ravel()))
            other_coefs = np.concatenate((other_coefs.ravel(), np.array(other.r).ravel()))
        difference = np.linalg.norm(self_coefs - other_coefs, ord=2)
        return difference



