import itertools
import math
from copy import deepcopy
import time

import utils

ids = ["111111111", "222222222"]
IMPASSABLE = "I"
GAS = "G"

RESET_PENALTY = 50
REFUEL_PENALTY = 10
DROP_IN_DESTINATION_REWARD = 100
INIT_TIME_LIMIT = 300
TURN_TIME_LIMIT = 0.1

# TODO CHECK CAPACITY FOR LARGER PROBLEMS
def dict_to_tuples(state):
    res = []
    # matrix
    if type(state) == list and state and type(state[0]) == list:
        r = []
        for l in state:
            r.append(tuple(l))
        return tuple(r)
    # reached the base case i.e. literal

    if type(state) != dict:
        if type(state) == list:
            return tuple(state)
        return state

    for key in sorted(list(state.keys())):
        res.append((key, dict_to_tuples(state[key])))
    return tuple(res)

def actions(state, matrix):
    taxis = state["taxis"]
    passengers = state["passengers"]
    rows = len(matrix)
    cols = len(matrix[0])
    acts = {}

    for taxi in taxis:
        acts[taxi] = []
        taxi_loc = x, y = state["taxis"][taxi]["location"]
        fuel = state["taxis"][taxi]["fuel"]
        if 0 <= x + 1 < rows and 0 <= y < cols and matrix[x + 1][y] != IMPASSABLE and fuel > 0:
            acts[taxi].append(("move", taxi, (x+1, y)))
        if 0 <= x - 1 < rows and 0 <= y < cols and matrix[x - 1][y] != IMPASSABLE and fuel > 0:
            acts[taxi].append(("move", taxi, (x - 1, y)))
        if 0 <= x < rows and 0 <= y + 1 < cols and matrix[x][y + 1] != IMPASSABLE and fuel > 0:
            acts[taxi].append(("move", taxi, (x, y + 1)))
        if 0 <= x < rows and 0 <= y - 1 < cols and matrix[x][y - 1] != IMPASSABLE and fuel > 0:
            acts[taxi].append(("move", taxi, (x, y - 1)))

        for passenger in passengers:
            psg_loc = state["passengers"][passenger]["location"]
            psg_dst = state["passengers"][passenger]["destination"]
            # Picking up
            if taxi_loc == psg_loc and state["taxis"][taxi]["capacity"] > 0 and psg_loc != psg_dst:
                acts[taxi].append(("pick up", taxi, passenger))
            if psg_loc == taxi and taxi_loc == psg_dst:
                acts[taxi].append(("drop off", taxi, passenger))

        acts[taxi].append(tuple(["wait", taxi]))

        if matrix[x][y] == GAS:
            acts[taxi].append(("refuel", taxi))

    res = []
    for action in itertools.product(*acts.values()):
        taxis_location_dict = dict([(t, state['taxis'][t]['location']) for t in state['taxis'].keys()])
        move_actions = [a for a in action if a[0] == 'move']
        for move_action in move_actions:
            taxis_location_dict[move_action[1]] = move_action[2]
        if len(set(taxis_location_dict.values())) != len(taxis_location_dict):
            break
        res.append(tuple(action))
    return res + ["reset", "terminate"]

class AbstractProblem:
    def __init__(self, an_input, init_state):
        """
        initiate the problem with the given input
        """
        self.initial_state = deepcopy(init_state)
        self.state = deepcopy(an_input)

        self.score = 0

    def result(self, action):
        """"
        update the state according to the action
        """
        self.apply(action)

    def apply(self, action):
        """
        apply the action to the state
        """
        if action == "reset":
            self.reset_environment()
            return
        if action == "terminate":
            return
        for atomic_action in action:
            self.apply_atomic_action(atomic_action)

    def apply_atomic_action(self, atomic_action):
        """
        apply an atomic action to the state
        """
        taxi_name = atomic_action[1]
        if atomic_action[0] == 'move':
            self.state['taxis'][taxi_name]['location'] = atomic_action[2]
            self.state['taxis'][taxi_name]['fuel'] -= 1
            return
        elif atomic_action[0] == 'pick up':
            passenger_name = atomic_action[2]
            self.state['taxis'][taxi_name]['capacity'] -= 1
            self.state['passengers'][passenger_name]['location'] = taxi_name
            return
        elif atomic_action[0] == 'drop off':
            passenger_name = atomic_action[2]
            self.state['passengers'][passenger_name]['location'] = self.state['taxis'][taxi_name]['location']
            self.state['taxis'][taxi_name]['capacity'] += 1
            self.score += DROP_IN_DESTINATION_REWARD
            return
        elif atomic_action[0] == 'refuel':
            self.state['taxis'][taxi_name]['fuel'] = self.initial_state['taxis'][taxi_name]['fuel']
            self.score -= REFUEL_PENALTY
            return
        elif atomic_action[0] == 'wait':
            return
        else:
            raise NotImplemented

    def reset_environment(self):
        """
        reset the state of the environment
        """
        self.state["taxis"] = deepcopy(self.initial_state["taxis"])
        self.state["passengers"] = deepcopy(self.initial_state["passengers"])
        self.score -= RESET_PENALTY
        return


def apply(state, action, init_state):
    def get_combinations(obj):
        "In this context it has all the passengers"
        res = []
        for L in range(len(obj) + 1):
            for subset in itertools.combinations(obj, L):
                yield subset
                #res.append(subset)
        #return res

    def _apply_goal_change(state):
        res = {}
        passengers = state["passengers"]
        for names in get_combinations(passengers):
            " The goal must change"
            prob = 1
            for passenger in state["passengers"]:
                chng_goal_prob = state["passengers"][passenger]["prob_change_goal"]
                len_goals = len(state["passengers"][passenger]["possible_goals"])
                if passenger in names:
                    prob *= chng_goal_prob * (1 / len_goals)
                else:
                    prob *= (1 - chng_goal_prob)

            if not names:
                _state = deepcopy(state)
                res[dict_to_tuples(_state)] = prob
            else:
                goals = {passenger: state["passengers"][passenger]["possible_goals"] for passenger in names}
                for c in itertools.product(*goals.values()):
                    _state = deepcopy(state)
                    for pasg, new_goal in zip(names, c):
                        _state["passengers"][pasg]["destination"] = new_goal
                    k = dict_to_tuples(_state)
                    if k in res:
                        res[k] += prob
                    else:
                        res[k] = prob
        a = [list(b) for b in res.items()]
        return a

    """
    Given state and action, returns the states 
    after applying the action and the probability to get that state
    NOTE:
    The states here differ in terms of goal changing and so
    """
    if action == "reset":
        return [(dict_to_tuples(deepcopy(init_state)), 1.0)]
    elif action == "terminate":
        return []

    new_state = deepcopy(state)

    for atmoic_action in action:
        act = atmoic_action[0]
        taxi = atmoic_action[1]

        if act == "move":
            # update taxis and passengers on the taxi's location
            new_state["taxis"][taxi]["location"] = atmoic_action[2]
            new_state["taxis"][taxi]["fuel"] -= 1
        elif act == "pick up":
            passenger = atmoic_action[2]
            new_state['taxis'][taxi]['capacity'] -= 1
            new_state["passengers"][passenger]["location"] = taxi
        elif act == "drop off":
            passenger = atmoic_action[2]
            new_state['taxis'][taxi]['capacity'] += 1
            new_state["passengers"][passenger]["location"] = new_state["passengers"][passenger]["destination"]
        elif act == "refuel":
            new_state["taxis"][taxi]["fuel"] = init_state["taxis"][taxi]["fuel"]

    res = _apply_goal_change(new_state)
    return res


def reward(state, action):
    r = 0
    if action is None:
        return 0
    if action == "reset":
        return -50

    for atomic_action in action:
        act = atomic_action[0]
        if act == "drop off":
            r += 100
        elif act == "refuel":
            r += -10

    return r


def create_all_states(state, matrix):
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

class OptimalTaxiAgent:
    def create_states(self):
        self.all_states = create_all_states(self.initial, self.matrix)
        return len(self.all_states)

    def __init__(self, initial, optimal=True):
        """
        When calling the constructor in an optimal context, Value Iterations will start automatically
        else the functions need to be called manually
        """
        start = time.perf_counter()
        self.initial = deepcopy(initial)
        self.turns = initial["turns to go"]
        self.matrix = self.initial["map"]

        del self.initial["turns to go"]
        del self.initial["optimal"]
        del self.initial["map"]

        if optimal:
            print("There are a total of ", self.create_states(), " states")
            self.value_iteration()
            print("finished VI")
            print("Value of first state: ", self.V[self.turns-1][dict_to_tuples(self.initial)])

            end = time.perf_counter()
            print("Runtime took: ", end - start)

        self.prev_action = None

    def value_iteration(self):
        """ V is a T * |S| matrix (list of dicts), where t is turns, |S| is the number of states"""
        self.V = [dict() for _ in range(self.turns)]
        self.PI = [dict() for _ in range(self.turns)]
        _acts = {dict_to_tuples(s): actions(s, self.matrix) for s in self.all_states}
        _apply = {tuple([dict_to_tuples(state), act]): apply(state, act, self.initial)
                  for state in self.all_states for act in actions(state, self.matrix)}
        #print("---", end - start)
        for t in range(self.turns):
            for s in self.all_states:
                max_val = -math.inf
                opt_act = None
                _s = dict_to_tuples(s)
                for act in _acts[_s]:
                    val = reward(s, act)
                    if t > 0:
                        val += sum([p * self.V[t-1][state] for state, p in _apply[tuple([_s, act])]])
                    if val > max_val:
                        max_val = val
                        opt_act = act

                self.V[t][_s] = max_val
                self.PI[t][_s] = opt_act

    def act(self, state):
        t = state["turns to go"] - 1
        temp_state = deepcopy(state)
        del temp_state["turns to go"]
        del temp_state["map"]
        del temp_state["optimal"]

        act = self.PI[t][dict_to_tuples(temp_state)]
        self.prev_action = act
        #print(act)
        return act


class TaxiAgent:
    def reduce_state(self, state):
        _state = deepcopy(state)

        if "passenger" in self.picked:
            psg = self.picked["passenger"]
            _state["passengers"] = {psg: state["passengers"][psg]}
        if "taxi" in self.picked:
            tx = self.picked["taxi"]
            _state["taxis"] = {tx: state["taxis"][tx]}
            self.one_taxi = False
        return _state

    def act_padding(self, act):
        res = []
        taxis = self.initial["taxis"]

        # If there's one taxi this condition won't be met and the original action will be returned
        if "taxi" in self.picked:
            # There's more than one taxi
            for taxi in taxis:
                if taxi == self.picked["taxi"]:
                    if isinstance(act, tuple):
                        res.append(act[0])
                        continue
                    return act
                else:
                    res.append(("wait", taxi))
        else:
            return act
        return tuple(res)

    def add_impassable(self, state):
        """
        If there's more than one taxis, we want to avoid a colision so we set the location of other taxis as impassable
        """
        _state = deepcopy(state)
        matrix = state["map"]
        for taxi in self.initial["taxis"]:
            if taxi != self.picked["taxi"]:
                x, y = self.initial["taxis"][taxi]["location"]
                matrix[x][y] = IMPASSABLE
        _state["map"] = matrix
        return _state

    def pick_taxi_passenger(self, chng_tx=False, chng_psg=False):
        def find_best_comb(state):
            def man_dist(a, b):
                return abs(a[0] - b[0]) + abs(a[1] - b[1])
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
            gas_stations = [(r, c) for r, _ in enumerate(matrix) for c, _ in enumerate(matrix[0]) if
                            matrix[r][c] == "G"]
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

        if not bool(self.picked):
            #d ict is empty
            t, p = find_best_comb(self.initial)
            self.picked["passenger"] = p
            self.picked["taxi"] = t
            # if len(self.psgs) > 1:
            #     self.picked["passenger"] = self.psgs.pop()
            # if len(self.txis) > 1:
            #     self.picked["taxi"] = self.txis.pop()
            return
        # If there are no taxis or passengers left, the dict will contain empty values as intended
        if chng_tx:
            self.picked["taxi"] = self.txis.pop()
        if chng_psg:
            self.picked["passenger"] = self.psgs.pop()

    def __init__(self, initial):
        print("\tNon optimal!!")
        self.initial = initial
        self.picked = {}
        self.one_taxi = True
        self.score = 0
        self.psgs = list(self.initial["passengers"].keys())
        self.txis = list(self.initial["taxis"].keys())
        self.pick_taxi_passenger()
        state = self.reduce_state(self.initial)
        #print(state)
        if not self.one_taxi:
            state = self.add_impassable(state)
        #print(state)
        self.optimalAgent = OptimalTaxiAgent(state)
        print("Finished initial")

    def act(self, state):
        reduced_state = self.reduce_state(state)
        act = self.act_padding(self.optimalAgent.act(reduced_state))
        self.score += reward(None, act)
        #print(self.score)
        #print(act)
        return act
