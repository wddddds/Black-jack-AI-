import Implementation_of_Easy21
import pylab
import random
import numpy as np

from Monte_Carlo_Control import load_obj, greedy
from collections import defaultdict
from Implementation_of_Easy21 import State


def convert_action_state_to_feature(state, action):
    feature = [0 for i in xrange(36)]
    if state.dealer >10 or state.dealer < 1 or state.player_sum > 21 or state.player_sum < 1:
        pass
    else:
        if state.dealer not in [4, 7]:
            d = [(state.dealer - 1) / 3 + 1]
        else:
            d = [(state.dealer - 1) / 3,(state.dealer - 1) / 3 + 1]

        if state.player_sum in [1, 2, 3]:
            p = [1]
        elif state.player_sum in [19, 20, 21]:
            p = [6]
        else:
            p = [(state.player_sum-1)/3, (state.player_sum-1)/3 + 1]

        for i in d:
            for j in p:
                feature[action*18+(i-1)*3+(j-1)] = 1
    feature = tuple(feature)

    return feature


def compute_MSE(avc, MC_result):
    MSE = 0
    count = 0
    for x in xrange(1,11):
        for y in xrange(1,22):
            for a in [0,1]:
                state = State(x, y)
                feature = convert_action_state_to_feature(state, a)
                # if (x, y, a) not in avc:
                #     avc[(x, y, a)] = 0

                MSE += (avc[feature] - MC_result[(x, y, a)]) ** 2
                count += 1

    MSE = MSE / count

    return MSE


def greedy_feature_version(action_value_function, feature_hit, feature_stick, epsilon):
    HIT = 0
    STICK = 1
    if random.random() > epsilon:
        if action_value_function[feature_hit] > action_value_function[feature_stick]:
            action = HIT
        elif action_value_function[feature_stick] > action_value_function[feature_hit]:
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


if __name__ == '__main__':

    MC_action_value_function = load_obj('action_value_function')

    parameter = [0 for i in xrange(36)]

    lambd = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    HIT = 0
    STICK = 1
    episode = 10000
    epsilon = 0.1
    alpha = 0.01

    MSE = []

    # implementing sarsa algorithm
    for lamb in lambd:

        if lamb == 0. or lamb == 1.:
            learning_curve = []

        # initialize value functions:
        action_value_function = defaultdict(float)
        for epi in xrange(episode):
            if epi % 100 == 0:
                if lamb == 0. or lamb == 1.:
                    learning_curve.append(compute_MSE(action_value_function, MC_action_value_function))

            state = Implementation_of_Easy21.State()
            feature_hit = convert_action_state_to_feature(state, HIT)
            feature_stick = convert_action_state_to_feature(state, STICK)
            eligibility_trace = np.array([0 for i in xrange(36)])
            player_sum = state.player_sum
            dealer = state.dealer

            # decide an action
            action = greedy_feature_version(action_value_function, feature_hit, feature_stick, epsilon)

            while state.terminal is False:
                if action == HIT:
                    eligibility_trace = np.add(eligibility_trace, feature_hit)
                else:
                    eligibility_trace = np.add(eligibility_trace, feature_stick)

                # update parameter
                state, reward = Implementation_of_Easy21.step(state, action)
                if reward is None:
                    reward = 0

                new_feature_hit = convert_action_state_to_feature(state, HIT)
                new_feature_stick = convert_action_state_to_feature(state, STICK)

                if action == HIT:
                    delta = reward - np.array(new_feature_hit).dot(parameter)
                else:
                    delta = reward - np.array(new_feature_stick).dot(parameter)

                # update action_value_function
                if action == HIT:
                    action_value_function[new_feature_hit] = np.array(new_feature_hit).dot(parameter)
                else:
                    action_value_function[new_feature_stick] = np.array(new_feature_stick).dot(parameter)

                # update delta, parameters, and eligibility-trace
                if action == HIT:
                    delta += action_value_function[new_feature_hit]
                else:
                    delta += action_value_function[new_feature_stick]

                parameter = np.add(parameter, alpha*delta*eligibility_trace)

                eligibility_trace = eligibility_trace * lamb

                action = greedy_feature_version(action_value_function, new_feature_hit,new_feature_stick,epsilon)
                feature_hit = new_feature_hit
                feature_stick = new_feature_stick

            print 'lambda: ', lamb, '\repisode:', epi,

        if lamb == 0. or lamb == 1.:
            learning_curve.append(compute_MSE(action_value_function, MC_action_value_function))

        if lamb == 0. or lamb == 1.:
            x = range(0, episode + 1, 100)
            print len(x)
            pylab.title('Learning curve of Mean-Squared Error against episode number: lambda = ' + str(lamb))
            pylab.xlabel("episode number")
            pylab.ylabel("Mean-Squared Error")
            pylab.plot(x, learning_curve)
            pylab.show()

        mse = compute_MSE(action_value_function, MC_action_value_function)
        MSE.append(mse)

    pylab.title('mean-squared error against lambda')
    pylab.xlabel("lambda")
    pylab.ylabel('MSE')
    pylab.plot(lambd, MSE)
    pylab.show()



