import problem_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn
import utils_io

###########
# problem 1
###########

def prob_1():
    epsilon = 0.1
    num_dims = 15
    num_samples = 1e7
    volumes = []
    fraction_of_circle_in_epsilon_skin = []

    for dim in range(1, num_dims + 1):
        samples = 2 * np.random.random([num_samples, dim])
        sample_norms = np.linalg.norm(samples, axis=1)
        within_circle = np.sum(np.where(sample_norms <= 1, 1, 0))
        volume_of_cube = 2 ** dim
        volume_of_circle = 4 * (within_circle / num_samples) * volume_of_cube
        volumes.append(volume_of_circle)
        within_epsilon_skin = np.sum(np.where((sample_norms <= 1) & (sample_norms > 1 - epsilon), 1, 0))
        fraction_of_circle_in_epsilon_skin.append(within_epsilon_skin / float(within_circle))

    vals = pd.Series(fraction_of_circle_in_epsilon_skin, index=range(1, num_dims + 1))  # for part 1b
    ax = vals.plot(title='2.1 Probability a point is within the epsilon-skin of a sphere')
    ax.set_xlabel('number of dimensions (may have too few samples for dim > 10)')
    ax.set_ylabel('P(point within ball is within epsilon-skin)')
    plt.ion()


def get_grid_has_inf_neighbor(grid):
    grid_filt = np.where(grid == 1, np.ones(grid.shape), np.zeros(grid.shape))
    shift_down = np.roll(grid_filt, 1, axis=0)
    shift_down[0, :] = 0.
    shift_up = np.roll(grid_filt, -1, axis=0)
    shift_up[-1, :] = 0.
    shift_right = np.roll(grid_filt, 1, axis=1)
    shift_right[:, 0] = 0.
    shift_left = np.roll(grid_filt, -1, axis=1)
    shift_left[:, -1] = 0.
    has_inf_neighbor = shift_down + shift_left + shift_up + shift_right
    return np.where(has_inf_neighbor > 0, np.ones(has_inf_neighbor.shape), np.zeros(has_inf_neighbor.shape))


def get_next_grid_val(val, random_var, transition_probabilities):
    return np.min(np.where(random_var <= transition_probabilities[val]))

def get_new_grid_with_protected():
    grid = np.zeros([10, 10])
    grid[1, 1] = 1
    grid[8, 8] = 1
    grid[1, 2] = 3
    grid[2, 3] = 3
    grid[2, 6] = 3
    grid[3, 5] = 3
    grid[5, 3] = 3
    grid[6, 2] = 3
    grid[7, 8] = 3
    grid[8, 9] = 3
    return grid


def get_new_grid():
    grid = np.zeros([10, 10])
    grid[1, 1] = 1
    grid[8, 8] = 1
    return grid

###########
# problem 3
###########

def prob_3():
    num_samples, transition_probabilities, T = problem_data.hw2_p3_data()
    final_grids = []
    num_deceased = []

    for sample in range(int(num_samples)):
        grids = []
        grid = get_new_grid_with_protected()
        for time in range(T):
            grids.append(grid)
            grid_has_inf_neighbor = get_grid_has_inf_neighbor(grid)
            transition_indices = (4 * grid_has_inf_neighbor + grid).ravel()
            next_state_vars = np.random.random([10, 10]).ravel()
            next_states = []
            for idx, transition_idx in enumerate(transition_indices):
                next_state_var = next_state_vars[idx]
                next_states.append(get_next_grid_val(transition_idx, next_state_var, transition_probabilities))
            grid = np.array(next_states).reshape(grid.shape)
        final_grids.append(grids[-1])
        num_deceased.append(len(np.where(grids[-1] == 2)[0]))
    utils_io.label('2.3')
    print 'Mean num deceased at T={0:d}, for {1:d} by {1:d} grid: '.format(T, int(num_samples)),\
        str(np.mean(num_deceased))

###########
# problem 5
###########

def prob_5a():
    pi, P, T = problem_data.hw2_p5_data()
    probs = np.zeros(T)
    for time in range(T):
        probs[time] = pi[0]
        pi = np.dot(pi, P)
    utils_io.label('2.5a')
    print 'p_T for T={0:d} equals: {1:f}'.format(T-1, np.mean(probs))


def get_random_idx(pdf):
    cdf = pdf.cumsum()
    rand = random.random()
    return np.min(np.where(rand <= cdf))


def prob_5b():
    num_samples = np.array([10, 100, 1000, 10000])
    avg_ones = np.zeros(len(num_samples))
    pi, P, T = problem_data.hw2_p5_data()
    for idx, num in enumerate(num_samples):
        ones = []
        for sample in range(num):
            number_of_ones = 0
            next_dist = pi
            for time in range(T):
                next_idx = get_random_idx(next_dist)
                if next_idx == 0:
                    number_of_ones += 1
                next_dist = P[next_idx]
            ones.append(number_of_ones)
        avg_ones[idx] = np.mean(ones) / T
    utils_io.label('2.5b')
    print 'num in sample: ', num_samples
    print 'p_T estimate for T={0:d} equals: {1:s}'.format(T-1, avg_ones)

if __name__ == '__main__':
    plt.show()
    prob_1()
    prob_3()
    prob_5a()
    prob_5b()
