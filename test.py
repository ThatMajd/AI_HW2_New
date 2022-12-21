import itertools
from copy import deepcopy
import ex2
import utils

state = {
        "optimal": False,
        "map": [['P', 'P', 'G'],
                ['P', 'P', 'P'],
                ['G', 'P', 'P']],
        "taxis": {'taxi 1': {"location": (0, 0), "fuel": 3, "capacity": 1},
                  'taxi 2': {"location": (0, 1), "fuel": 3, "capacity": 1}},
        "passengers": {'Dana': {"location": (0, 2), "destination": (0, 0),
                                "possible_goals": ((1, 1), (2, 2)), "prob_change_goal": 0.1},
                       'Dan': {"location": (2, 0), "destination": (0, 0),
                               "possible_goals": ((1, 1), (2, 2)), "prob_change_goal": 0.1}
                       },
        "turns to go": 100
    }


def get_combinations(obj):
    "In this context it has all the passengers"
    res = []
    for L in range(len(obj) + 1):
        for subset in itertools.combinations(obj, L):
            res.append(subset)
    return res


def _a(state, names):
    " The goal must change"
    res = []
    prob = 1
    for passenger in state["passengers"]:
        chng_goal_prob = state["passengers"][passenger]["prob_change_goal"]
        if passenger in names:
            prob *= chng_goal_prob
        else:
            prob *= (1 - chng_goal_prob)
    if not names:
        _state = deepcopy(state)
        return [tuple([_state, prob])]
    _state = deepcopy(state)
    goals = {passenger: state["passengers"][passenger]["possible_goals"] for passenger in names}
    for c in itertools.product(*goals.values()):
        _state = deepcopy(state)
        for pasg, new_goal in zip(names, c):
            _state["passengers"][pasg]["destination"] = new_goal
        res.append(tuple([_state, prob]))
    return res


def _p(state):
    res = []
    _state = deepcopy(state)
    passengers = state["passengers"]
    for sub_pasg in get_combinations(passengers):
        res += _a(state, sub_pasg)
    return res

state = {'optimal': True,
         'map': [['P', 'P', 'P'],
                 ['P', 'G', 'P'],
                 ['P', 'P', 'P']],
         'taxis': {'taxi 1': {'location': (2, 0), 'fuel': 5, 'capacity': 1}},
         'passengers': {'Dana': {'location': (2, 2), 'destination': (0, 0), 'possible_goals': ((1, 1), (2, 2), (3, 3)), 'prob_change_goal': 0.1}},
         'turns to go': 100}

a = [["A", 0.5], ["B", 0.1], ["C", 0.9]]


def normalize_probs(list_states):
    res = list_states
    div = sum([p for _, p in res])
    for i, _ in enumerate(res):
        res[i][1] /= div
    return res


a = normalize_probs(a)
print(a)
