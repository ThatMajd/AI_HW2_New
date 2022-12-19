import itertools
from copy import deepcopy

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
        return [_state, prob]
    _state = deepcopy(state)
    goals = {passenger: state["passengers"][passenger]["possible_goals"] for passenger in names}
    for c in itertools.product(*goals.values()):
        _state = deepcopy(state)
        for pasg, new_goal in zip(names, c):
            _state["passengers"][pasg]["destination"] = new_goal
        res.append((_state, prob))
    return res


def _p(state):
    res = []
    _state = deepcopy(state)
    passengers = state["passengers"]
    for sub_pasg in get_combinations(passengers):
        res += _a(state, sub_pasg)
    return res

temp =  _p(state)
print(temp[1])

