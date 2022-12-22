import itertools
from copy import deepcopy
import ex2
import utils

state = {
        "optimal": True,
        "map": [['P', 'P', 'P'],
                ['P', 'G', 'P'],
                ['P', 'P', 'P']],
        "taxis": {'taxi 1': {"location": (0, 0), "fuel": 10, "capacity": 1}},
        "passengers": {'Dana': {"location": (2, 2), "destination": (0, 0),
                                "possible_goals": ((0, 0), (2, 2)), "prob_change_goal": 0.1}}
    }


#print(type(state["passengers"]["Dana"]["location"]))
#sts = ex2.create_all_states(state)

#bad = {'optimal': True, 'map': [['P', 'P', 'P'], ['P', 'G', 'P'], ['P', 'P', 'P']], 'taxis': {'taxi 1': {'location': (0, 0), 'fuel': 10, 'capacity': 1}}, 'passengers': {'Dana': {'location': (2, 2), 'destination': (0, 0), 'possible_goals': ((1, 1), (2, 2)), 'prob_change_goal': 0.1}}}

a = {"A": 2, "B": 2}
#print([list(b) for b in a.items()])
for a in ex2.apply(state, [("move", "taxi 1", (1, 0))]):
    print(a)
