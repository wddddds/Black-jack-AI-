import Implementation_of_Easy21
import pylab

from Monte_Carlo_Control import load_obj, greedy
from collections import defaultdict


def compute_MSE(avc, MC_result):
    MSE = 0
    count = 0
    for x in xrange(1,11):
        for y in xrange(1,22):
            for a in [0,1]:
                if (x, y, a) not in avc:
                    avc[(x, y, a)] = 0
                MSE += (avc[(x, y, a)] - MC_result[(x, y, a)]) ** 2
                count += 1

    MSE = MSE / count

    return MSE

if __name__ == '__main__':

    MC_action_value_function = load_obj('action_value_function')

    lambd = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    HIT = 0
    STICK = 1

    episode = 10000
    N_zero = 10

    MSE = []

    # implementing sarsa algorithm
    for lamb in lambd:

        if lamb == 0. or lamb == 1.:
            learning_curve = []

        # initialize value functions:
        action_value_function = defaultdict(float)
        action_value_set = defaultdict(int)
        value_function = defaultdict(float)
        num_action_state = defaultdict(int)
        for epi in xrange(episode):
            if epi % 100 == 0:
                if lamb == 0. or lamb == 1.:
                    learning_curve.append(compute_MSE(action_value_function, MC_action_value_function))

            state = Implementation_of_Easy21.State()
            eligibility_trace = defaultdict(int)
            eligibility_trace_set = defaultdict(int)
            player_sum = state.player_sum
            dealer = state.dealer

            # define epsilon
            if (dealer, player_sum) not in value_function:
                min_action_num = 0
            else:
                min_action_num = min((x for x in num_action_state if x[0] == dealer and x[1] == player_sum),
                                     key=lambda l: l[2])[2]
            epsilon = 1.0 * N_zero / (N_zero + min_action_num)

            # decide an action
            action = greedy(action_value_function, state, epsilon)

            while state.terminal is False:
                num_action_state[(dealer, player_sum, action)] += 1
                # update parameter
                state, reward = Implementation_of_Easy21.step(state, action)
                if reward is None:
                    reward = 0

                # update the dealer and player sum
                dealer_next = state.dealer
                player_sum_next = state.player_sum

                # update the epsilon
                if (dealer_next, player_sum_next) not in value_function:
                    min_action_num = 0
                else:
                    min_action_num = min((x for x in num_action_state if x[0] == dealer_next and x[1] == player_sum_next),
                                         key=lambda l: l[2])[2]
                epsilon = 1.0 * N_zero / (N_zero + min_action_num)

                # take new actions
                action_next = greedy(action_value_function, state, epsilon)

                # update alpha, delta and eligibility_trace

                alpha = 1.0 / num_action_state[(dealer, player_sum, action)]

                if (dealer_next, player_sum_next, action_next) not in action_value_set:
                    action_value_function[(dealer_next, player_sum_next, action_next)] = 0
                    action_value_set[(dealer_next, player_sum_next, action_next)] = 1
                if (dealer, player_sum, action) not in action_value_set:
                    action_value_function[(dealer, player_sum, action)] = 0
                    action_value_set[(dealer, player_sum, action)] = 1

                delta = reward + action_value_function[(dealer_next, player_sum_next, action_next)] \
                        - action_value_function[(dealer, player_sum, action)]

                if (dealer_next, player_sum_next, action_next) not in eligibility_trace_set:
                    eligibility_trace[(dealer_next, player_sum_next, action_next)] = 0
                    eligibility_trace_set[(dealer_next, player_sum_next, action_next)] = 1
                eligibility_trace[(dealer, player_sum, action)] += 1

                # update action_value_function and eligibility_trace
                action_value_function = {k: action_value_function.get(k, 0) + alpha*delta*eligibility_trace.get(k, 0)
                                         for k in set(action_value_function)}
                eligibility_trace = {k: lamb * eligibility_trace.get(k, 0) for k in set(eligibility_trace)}

                # update state and action
                dealer = dealer_next
                player_sum = player_sum_next
                action = action_next

            print 'lambda: ', lamb, '\repisode:', epi,

        if lamb == 0. or lamb == 1.:
            learning_curve.append(compute_MSE(action_value_function, MC_action_value_function))

        if lamb == 0. or lamb == 1.:
            x = range(0, episode + 1, 100)
            print len(x)
            pylab.title('Mean-Squared Error against episode: lambda = ' + str(lamb))
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



