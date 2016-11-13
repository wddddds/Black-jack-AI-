import random
import collections


def draw():
    p = random.random()
    if p <= 2.0/3:
        return random.randint(1, 10)
    else:
        return -random.randint(1, 10)


def step(state, action):
    HIT = 0
    STICK = 1

    def check_burst(score):
        if 1 <= score <= 21:
            return False
        else:
            return True

    player_burst = False
    dealer_burst = False

    if action == HIT:
        state.player_sum += draw()
        player_burst = check_burst(state.player_sum)

        if player_burst:
            state.terminal = True

    elif action == STICK:
        state.dealer += draw()
        dealer_burst = check_burst(state.dealer)

        while dealer_burst is False and state.dealer < 17:
            state.dealer += draw()
            dealer_burst = check_burst(state.dealer)

        state.terminal = True

    reward = None

    if state.terminal:
        if dealer_burst:
            reward = 1
        elif player_burst:
            reward = -1
        else:
            if state.player_sum > state.dealer:
                reward = 1
            elif state.dealer > state.player_sum:
                reward = -1
            else:
                reward = 0

    return state, reward


class State:
    def __init__(self, dealer=None, player_sum=None):
        if dealer is not None and player_sum is not None:
            self.dealer = dealer
            self.player_sum = player_sum
        elif dealer is None and player_sum is None:
            self.dealer = random.randint(1, 10)
            self.player_sum = random.randint(1, 10)
    terminal = False

    def __str__(self):
        return '''State:
        dealer: %s
        player: %s
        terminal :%s''' % (self.dealer, self.player_sum, self.terminal)


if __name__ == '__main__':

    Counter = collections.Counter

    # Q1 part1, check the draw function.
    checkDraw = []
    print 'checkDraw:'
    for i in xrange(1000):
        checkDraw.append(draw())

    counter = Counter(checkDraw)
    for k, v in counter.iteritems():
        if k < 0:
            color = -1
        else:
            color = 1
        if k > 0:
            print k, color, 1.0*v/1000
        else:
            print -k, color, 1.0*v/1000

    # Q1 part2, check the setp function.
    print 'checkStep:'
    HIT = 0
    STICK = 1

    checkStep = []
    for i in xrange(1000):
        s = State(10, 15)
        s, r = step(s, STICK)
        new_data = (s.dealer, s.player_sum, r)
        checkStep.append(new_data)

    counter = Counter(checkStep)
    for k, v in counter.iteritems():
        print k[0], k[1], k[2], 1.0*v/1000


