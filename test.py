from Linear_Function_Approximation import convert_action_state_to_feature
from Implementation_of_Easy21 import State


HIT = 0
STICK = 1

state = State()
convert_action_state_to_feature(state, HIT)
for i in xrange(10):
    state = State()
    print state
    print convert_action_state_to_feature(state, HIT)
    print convert_action_state_to_feature(state, STICK)