import itertools
import math
from copy import deepcopy
import time
ids = ["111111111", "222222222"]
IMPASSABLE = "I"

RESET_PENALTY = 50
REFUEL_PENALTY = 10
DROP_IN_DESTINATION_REWARD = 100
INIT_TIME_LIMIT = 300
TURN_TIME_LIMIT = 0.1

# TODO CHECK CAPACITY FOR LARGER PROBLEMS

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

        if matrix[x][y] == 'G':
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
    return res

class AbstractProblem:
    def __init__(self, an_input):
        """
        initiate the problem with the given input
        """
        self.initial_state = deepcopy(an_input)
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


def apply(state, action):
    def get_combinations(obj):
        "In this context it has all the passengers"
        res = []
        for L in range(len(obj) + 1):
            for subset in itertools.combinations(obj, L):
                res.append(subset)
        return res

    def _apply_goal_change(state, names):
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
            #_state["turns to go"] -= 1
            return [[_state, prob]]
        goals = {passenger: state["passengers"][passenger]["possible_goals"] for passenger in names}
        for c in itertools.product(*goals.values()):
            _state = deepcopy(state)
            for pasg, new_goal in zip(names, c):
                _state["passengers"][pasg]["destination"] = new_goal
            #_state["turns to go"] -= 1
            res.append([_state, prob])
        return res

    def _p(state):
        res = []
        _state = deepcopy(state)
        passengers = state["passengers"]
        for sub_pasg in get_combinations(passengers):
            res += _apply_goal_change(state, sub_pasg)
        return res

    def normalize_probs(list_states):
        res = list_states
        div = sum([p for _, p in res])
        for i, _ in enumerate(res):
            res[i][1] /= div
        return res

    """
    Given state and action, returns the states 
    after applying the action and the probability to get that state
    NOTE:
    The states here differ in terms of goal changing and so
    """

    t = AbstractProblem(state)
    t.apply(action)
    new_state = t.state
    del t
    if action == "reset":
        return [(new_state, 1.0)]
    elif action == "terminate":
        return []
    res = _p(new_state)
    # TODO check how state with goal changed but same destination should be dealt with
    return normalize_probs(res)


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

    res = []

    for taxi in state["taxis"]:
        for taxi_loc in _possible_tiles(n, m, matrix):
            for f in range(state["taxis"][taxi]["fuel"] + 1):
                for passenger in state["passengers"]:
                    all_psg_locs = list(set(
                        [state["passengers"][passenger]["location"]] + list(state["taxis"].keys())
                        + list(state["passengers"][passenger]["possible_goals"])))
                    for psg_loc in all_psg_locs:
                        for dst in set(tuple([state["passengers"][passenger]["destination"]]) +
                                       state["passengers"][passenger]["possible_goals"]):
                            st = deepcopy(state)
                            st["taxis"][taxi]["location"] = taxi_loc
                            st["taxis"][taxi]["fuel"] = f
                            st["passengers"][passenger]["location"] = psg_loc
                            st["passengers"][passenger]["destination"] = dst
                            num_in_taxi = {}
                            for psg in st["passengers"]:
                                p_loc = st["passengers"][psg]["location"]
                                #print(p_loc, type(p_loc) == str)
                                if type(p_loc) is str:
                                    if p_loc not in num_in_taxi:
                                        num_in_taxi[p_loc] = 0
                                    num_in_taxi[p_loc] += 1
                            for t in num_in_taxi.keys():
                                st["taxis"][t]["capacity"] -= num_in_taxi[t]
                            res.append(st)
    return res


class OptimalTaxiAgent:
    def __init__(self, initial):
        start = time.perf_counter()
        self.initial = deepcopy(initial)
        self.turns = initial["turns to go"]
        del self.initial["turns to go"]

        matrix = self.initial["map"]
        del self.initial["optimal"]
        del self.initial["map"]
        self.all_states = create_all_states(self.initial, matrix)
        self.value_iteration(matrix)
        print("finished VI")
        print("Value of first state: ", self.V[self.turns-1][str(self.initial)])
        end = time.perf_counter()
        print("Runtime took: ", end - start)
        # file = open("policy.pkl", "wb")
        # pickle.dump(self.PI, file)
        # file.close()
        self.prev_action = None

    def value_iteration(self, matrix):
        """ V is a T * |S| matrix (list of dicts), where t is turns, |S| is the number of states"""

        self.V = [dict() for _ in range(self.turns)]
        self.PI = [dict() for _ in range(self.turns)]
        for t in range(self.turns):
            for s in self.all_states:
                max_val = -math.inf
                opt_act = None
                for act in actions(s, matrix):
                    val = reward(s, act)
                    if t > 0:
                        val += sum([p * self.V[t-1][str(state)] for state, p in apply(s, act)])
                    if val > max_val:
                        max_val = val
                        opt_act = act
                self.V[t][str(s)] = max_val
                self.PI[t][str(s)] = opt_act

    def act(self, state):
        def _dropped_off_all(action):
            if action is None:
                return False
            for atom_action in action:
                if atom_action[0] != "drop off":
                    return False
            return True

        if _dropped_off_all(self.prev_action):
            self.prev_action = "reset"
            return "reset"

        t = state["turns to go"] - 1
        temp_state = deepcopy(state)
        del temp_state["turns to go"]
        del temp_state["map"]
        del temp_state["optimal"]

        act = self.PI[t][str(temp_state)]
        self.prev_action = act
        print(act)
        return act


class TaxiAgent:
    def __init__(self, initial):
        self.initial = initial

    def act(self, state):
        raise NotImplemented
