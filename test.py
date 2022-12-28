import itertools
import time
from copy import deepcopy
import ex2
import utils
import json

# state = {
#         'optimal': True,
#         "turns to go": 50,
#         'map': [['P', 'P', 'P', 'P', 'G'], ],
#         'taxis': {'taxi 1': {'location': (0, 0), 'fuel': 1, 'capacity': 1},
#                   'taxi 2': {'location': (0, 0), 'fuel': 2, 'capacity': 1}},
#         'passengers': {'Michael': {'location': (2, 2), 'destination': (0, 0),
#                                    "possible_goals": ((0, 1), (0, 4)), "prob_change_goal": 0.7},
#                        'Din': {'location': (2, 2), 'destination': (0, 0),
#                                "possible_goals": ((0, 1), (0, 4)), "prob_change_goal": 0.2},
#                        }
#     }
state = {
        "optimal": True,
        "map": [['P', 'P', 'P'],
                ['P', 'G', 'P'],
                ['P', 'P', 'P-']],
        "taxis": {'taxi 1': {"location": (0, 0), "fuel": 10, "capacity": 1}},
        "passengers": {'Dana': {"location": (2, 2), "destination": (0, 0),
                                "possible_goals": ((0, 0), (2, 2)), "prob_change_goal": 0.1}},
        "turns to go": 100
    }
def create_all_states_2(state, matrix):
    def _possible_tiles(n,m, matrix):
        res = []
        for i in range(n):
            for j in range(m):
                if matrix[i][j] != 'I':
                    res.append((i, j))
        return res
    """
    Generates all possible states, assumes state given has no 'turns to go' key
    """
    n = len(matrix)
    m = len(matrix[0])

    # Taxis
    a = [(taxi, state["taxis"][taxi]["fuel"]) for taxi in state["taxis"]]
    comb_taxis = {}
    for t, fuel in a:
        comb_taxis[t] = []
        for f in range(fuel+1):
            for loc in _possible_tiles(n, m, matrix):
                comb_taxis[t].append([f, loc])
    comb_passengers = {}
    for passenger in state["passengers"]:
        comb_passengers[passenger] = []
        all_psg_locs = list(set(
            [state["passengers"][passenger]["location"]] + [state["passengers"][passenger]["destination"]]
            + list(state["taxis"].keys())
            + list(state["passengers"][passenger]["possible_goals"])
        ))
        for loc in all_psg_locs:
            for dst in set(tuple([state["passengers"][passenger]["destination"]]) +
                                       state["passengers"][passenger]["possible_goals"]):
                comb_passengers[passenger].append([loc, dst])
    res = []
    for t in itertools.product(*comb_passengers.values()):
        for k in itertools.product(*comb_taxis.values()):
            s = {}
            s["passengers"] = {}
            s["taxis"] = {}
            num_in_taxi = {taxi: state["taxis"][taxi]["capacity"] for taxi in state["taxis"]}
            for passenger, (_loc, _dst) in zip(state["passengers"], t):
                if isinstance(_loc, str):
                    num_in_taxi[_loc] -= 1
                s["passengers"][passenger] = {
                    "location": _loc,
                    "destination": _dst,
                    'possible_goals': state["passengers"][passenger]["possible_goals"],
                    'prob_change_goal': state["passengers"][passenger]["prob_change_goal"],
                }
            for taxi, (_f, _loc) in zip(state["taxis"], k):
                s["taxis"][taxi] = {
                    'location': _loc, 'fuel': _f, 'capacity': num_in_taxi[taxi]
                }
            res.append(s)
    return res

print(len(create_all_states_2(state, state["map"])))
for a in create_all_states_2(state, state["map"]):
    print(a)
