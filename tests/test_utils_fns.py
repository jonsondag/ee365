import sys

sys.path.append('../')

import nose
import numpy as np
from nose.tools import assert_almost_equal, assert_equal, assert_greater, assert_true
from stochastic_control_class import utils_fns
import unittest


class TestFunctionUtils(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        n = 3
        m = 1

        self.P = np.random.random([n, n])
        self.q = np.random.random([n, m])
        self.r = np.random.random()

    def test_to_quad_mat(self):
        quad_mat = utils_fns.to_quad_mat(self.P, self.q, self.r)
        quad_mat_manual = np.vstack(
            [np.hstack([self.P, self.q]), np.hstack([self.q.T, np.array(self.r).reshape([1, 1])])])
        assert_true(np.array_equal(quad_mat, quad_mat_manual))

    def test_from_quad_mat(self):
        quad_mat = np.vstack([np.hstack([self.P, self.q]), np.hstack([self.q.T, np.array(self.r).reshape([1, 1])])])
        P, q, r = utils_fns.from_quad_mat(quad_mat)
        assert_true(np.array_equal(self.P, P))
        assert_true(np.array_equal(self.q, q))
        assert_equal(self.r, r)


class TestLinearFunction(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        n = 3
        m = 1

        self.A_1 = np.random.random([n, n])
        self.b_1 = np.random.random([n, m])
        self.A_2 = np.random.random([n, n])
        self.b_2 = np.random.random([n, m])

        self.lf_1 = utils_fns.LinearFunction(self.A_1, self.b_1)
        self.lf_2 = utils_fns.LinearFunction(self.A_2, self.b_2)

    def test_plus_linear(self):
        lf_sum = self.lf_1.plus_linear(self.lf_2)

        assert_true(np.array_equal(lf_sum.A, self.lf_1.A + self.lf_2.A))
        assert_true(np.array_equal(lf_sum.b, self.lf_1.b + self.lf_2.b))

    def test_times_constant(self):
        constant = 5

        lf_times_contant = self.lf_1.times_constant(5)
        assert_true(np.array_equal(lf_times_contant.A, self.lf_1.A * constant))
        assert_true(np.array_equal(lf_times_contant.b, self.lf_1.b * constant))

    def test_compose(self):
        lf_composed = self.lf_1.compose(self.lf_2)
        assert_true(np.array_equal(lf_composed.A, np.dot(self.lf_1.A, self.lf_2.A)))
        assert_true(np.array_equal(lf_composed.b, self.lf_1.b + np.dot(self.lf_1.A, self.lf_2.b)))


class TestQuadraticFunction(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.n = 3
        self.m = 1

        self.P = np.random.random([self.n, self.n])
        self.P[self.m:, self.m:] = np.dot(self.P[self.m:, self.m:].T, self.P[self.m:, self.m:])  # ensure P_uu is PSD
        self.q = np.random.random([self.n, self.m])
        self.r = np.random.random()

    def test_partial_minimization(self):
        quad_func = utils_fns.QuadraticFunction(self.P, self.q, self.r)
        qfx, lfu = quad_func.partial_minimization(2)

        x = np.array([0.0]).reshape([1, 1])
        min_val = qfx.evaluate(x)
        P_uu = self.P[self.m:, self.m:]
        P_ux = self.P[self.m:, 0:self.m]
        q_u = self.q[self.m:]
        min_u = -np.dot(np.linalg.inv(P_uu), (np.dot(P_ux, x) + q_u))

        min_input = np.vstack([x, min_u])
        val_at_min_input = quad_func.evaluate(min_input)

        assert_almost_equal(min_val, val_at_min_input)

        rand = np.vstack([x, 0.01 * np.random.random(len(min_u))[:, np.newaxis]])

        almost_min_input_plus = min_input + rand
        print quad_func.evaluate(almost_min_input_plus)
        print val_at_min_input
        assert_true(quad_func.evaluate(almost_min_input_plus) > val_at_min_input)

        almost_min_input_minus = min_input - rand
        print quad_func.evaluate(almost_min_input_plus)
        print val_at_min_input
        assert_true(quad_func.evaluate(almost_min_input_minus) > val_at_min_input)

    def test_precompose_linear(self):
        quad_func = utils_fns.QuadraticFunction(self.P, self.q, self.r)
        A = np.random.random([self.n, self.n])
        b = np.random.random([self.n, 1])
        lin_func = utils_fns.LinearFunction(A, b)
        x = np.random.random([self.n, 1])

        direct_result = quad_func.evaluate(lin_func.evaluate(x))
        precompose_result = quad_func.precompose_linear(lin_func).evaluate(x)

        assert_almost_equal(direct_result, precompose_result)
