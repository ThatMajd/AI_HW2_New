import itertools
import time
from copy import deepcopy
import ex2
import utils
import json

state = {
        "optimal": True,
        "map": [['P', 'P', 'P'],
                ['P', 'G', 'P'],
                ['P', 'P', 'P']],
        "taxis": {'taxi 1': {"location": (0, 0), "fuel": 10, "capacity": 1}},
        "passengers": {'Dana': {"location": (2, 2), "destination": (0, 0),
                                "possible_goals": ((0, 0), (2, 2)), "prob_change_goal": 0.1}}
    }


a = {'taxis': {'taxi 1': {'location': (0, 0), 'fuel': 0, 'capacity': 1}}, 'passengers': {'Dana': {'location': (0, 0), 'destination': (2, 2), 'possible_goals': ((0, 0), (2, 2)), 'prob_change_goal': 0.1}}}

def tuples_to_dict(T):
    dictionary = {}
    for key, value in T:
        if isinstance(value, tuple) and not isinstance(value[0], int) and not key == 'possible_goals':
            value = tuples_to_dict(value)
        dictionary[key] = value
    return dictionary



#print(tup_a)
start = time.perf_counter()
print(deepcopy(a))
end = time.perf_counter()
print(end-start)
