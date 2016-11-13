import Implementation_of_Easy21
import random
import numpy as np
import pylab
import pickle

from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict


def greedy(action_value_function, state, epsilon):
    dealer = state.dealer
    player_sum = state.player_sum
    HIT = 0
    STICK = 1
    if random.random() > epsilon:
        if action_value_function[(dealer, player_sum, HIT)] > \
                action_value_function[(dealer, player_sum, STICK)]:
            action = HIT
        elif action_value_function[(dealer, player_sum, STICK)] > \
                action_value_function[(dealer, player_sum, HIT)]:
            action = STICK
        else:
            if random.random() > 0.5:
                action = HIT
            else:
                action = STICK
    else:
        if random.random() > 0.5:
            action = HIT
        else:
            action = STICK

    return action


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':

    HIT = 0
    STICK = 1

    action_value_function = defaultdict(float)
    value_function = defaultdict(float)
    num_action_state = defaultdict(int)

    N_zero = 10
    num_episode = 1000000

    for epi in xrange(num_episode):

        state = Implementation_of_Easy21.State()
        while state.terminal is False:
            dealer = state.dealer
            player_sum = state.player_sum

            if (dealer, player_sum) not in value_function:
                min_action_num = 0
            else:
                min_action_num = \
                    min((x for x in num_action_state if x[0] == dealer and x[1] == player_sum), key=lambda l: l[2])[2]
            epsilon = 1.0 * N_zero / (N_zero + min_action_num)

            # decide the action.
            action = greedy(action_value_function, state, epsilon)

            # update parameter
            num_action_state[(dealer, player_sum, action)] += 1
            alpha = 1.0 / num_action_state[(dealer, player_sum, action)]

            state, reward = Implementation_of_Easy21.step(state, action)
            if reward is None:
                reward = 0

            # update action value function and thus value function:
            action_value_function[(dealer, player_sum, action)] += \
                alpha * (reward - action_value_function[(dealer, player_sum, action)])

            if action_value_function[(dealer, player_sum, HIT)] > action_value_function[(dealer, player_sum, STICK)]:
                value_function[(dealer, player_sum)] = action_value_function[(dealer, player_sum, HIT)]
            else:
                value_function[(dealer, player_sum)] = action_value_function[(dealer, player_sum, STICK)]

        if epi % 1000 == 0:
            print '\repisode:', epi,

    # plot the optimal value function
    x = range(1, 11)
    y = range(1, 22)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[0.0 for i in range(len(x))] for j in range(1, 22)])
    for i in x:
        for j in y:
            Z[j - 1][i - 1] = value_function[(i, j)]
    fig = pylab.figure()
    ax = Axes3D(fig)
    pylab.title("Optimal Value Function (Monte-Carlo Control)")
    ax.set_xlabel("Dealer Showing")
    pylab.xlim([1, 10])
    pylab.xticks(range(1, 11))
    ax.set_ylabel("Player Sum")
    pylab.ylim([1, 21])
    pylab.yticks(range(1, 22))
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
    pylab.show()

    # checkQ
    for i in x:
        for j in y:
            for a in [0, 1]:
                print i, j, a, action_value_function[(i, j, a)]

    # save result
    save_obj(action_value_function, 'action_value_function')
    save_obj(value_function, 'value_function')
    save_obj(num_action_state, 'num_action_state')
