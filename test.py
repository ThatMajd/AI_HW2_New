import itertools
import math
import time
from copy import deepcopy
import ex2
import utils
import json
import networkx as nx
import matplotlib.pyplot as plt

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
        'optimal': False,
        "turns to go": 100,
        'map': [['P', 'P', 'P', 'P', 'P'],
                ['P', 'I', 'P', 'G', 'P'],
                ['P', 'P', 'I', 'P', 'P'],
                ['P', 'P', 'P', 'I', 'P']],
        'taxis': {'taxi 1': {'location': (2, 0), 'fuel': 5, 'capacity': 2},
                  'taxi 2': {'location': (0, 1), 'fuel': 6, 'capacity': 2}},
        'passengers': {'Iris': {'location': (0, 0), 'destination': (1, 4),
                                'possible_goals': ((1, 4),), 'prob_change_goal': 0.2},
                       'Daniel': {'location': (3, 1), 'destination': (2, 1),
                                  'possible_goals': ((2, 1), (0, 1), (3, 1)), 'prob_change_goal': 0.2},
                       'Freyja': {'location': (2, 3), 'destination': (2, 4),
                                  'possible_goals': ((2, 4), (3, 0), (3, 2)), 'prob_change_goal': 0.2},
                       'Tamar': {'location': (3, 0), 'destination': (3, 2),
                                 'possible_goals': ((3, 2),), 'prob_change_goal': 0.2}},
    }
def man_dist(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def find_best_comb(state):
    def best_gas_station(loc1, loc2, gas, f):
        _min = math.inf
        best_gas = None
        for g in gas:
            d1 = man_dist(loc1, g)
            d2 = man_dist(g, loc2)
            if d1 < f or d2 < f:
                continue
            if d1 + d2 < _min:
                _min = d1 + d2
                best_gas = g
        return best_gas, _min


    total = math.inf
    taxis = state["taxis"]
    passengers = state["passengers"]
    matrix = state["map"]
    gas_stations = [(r, c) for r, _ in enumerate(matrix) for c, _ in enumerate(matrix[0]) if matrix[r][c] == "G"]
    res = None
    for taxi in taxis:
        moves = 0
        taxi_loc = taxis[taxi]["location"]
        fuel = taxis[taxi]["fuel"]
        "Find closest passenger without refuel"
        d1 = math.inf
        closest_psg = None
        for passenger in passengers:
            psg_loc = passengers[passenger]["location"]
            temp = man_dist(taxi_loc, psg_loc)
            # TODO check whether <= is needed
            if temp < fuel and temp < d1:
                d1 = temp
                closest_psg = passenger

        if closest_psg is None:
            assert d1 is math.inf
            for passenger in passengers:
                gas, distance = best_gas_station(taxi_loc, passengers[passenger], gas_stations, fuel)
                if d1 > distance:
                    d1 = distance

        if d1 is math.inf:
            continue
        moves += d1

        if d1 > fuel:
            fuel -= d1 - fuel
        else:
            fuel -= d1

        destinations = set(tuple([state["passengers"][closest_psg]["destination"]])
                           + state["passengers"][closest_psg]["possible_goals"])
        psg_loc = passengers[closest_psg]["location"]
        d2 = math.inf
        furthest = None
        for dst in destinations:
            temp = man_dist(psg_loc, dst)
            if temp < d2 and temp < fuel:
                d2 = temp
                furthest = dst
        if furthest is None:
            assert d2 is math.inf
            for dst in destinations:
                gas, distance = best_gas_station(psg_loc, dst, gas_stations, fuel)
                if d2 > distance:
                    d2 = distance
        if d2 is math.inf:
            continue
        moves += d2

        if moves < total:
            total = moves
            res = (taxi, closest_psg)
    return res

print(find_best_comb(state))