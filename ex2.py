import itertools
from copy import deepcopy

ids = ["111111111", "222222222"]
IMPASSABLE = "I"

RESET_PENALTY = 50
REFUEL_PENALTY = 10
DROP_IN_DESTINATION_REWARD = 100
INIT_TIME_LIMIT = 300
TURN_TIME_LIMIT = 0.1

def actions(state):
    def _is_pick_up_action_legal(pick_up_action):
        taxi_name = pick_up_action[1]
        passenger_name = pick_up_action[2]
        # check same position
        if state['taxis'][taxi_name]['location'] != state['passengers'][passenger_name]['location']:
            return False
        # check taxi capacity
        if state['taxis'][taxi_name]['capacity'] <= 0:
            return False
        # check passenger is not in his destination
        if state['passengers'][passenger_name]['destination'] == state['passengers'][passenger_name]['location']:
            return False
        return True

    def _is_drop_action_legal(drop_action):
        taxi_name = drop_action[1]
        passenger_name = drop_action[2]
        # check same position
        if state['taxis'][taxi_name]['location'] != state['passengers'][passenger_name]['destination']:
            return False
        return True

    def _is_refuel_action_legal(refuel_action):
        """
        check if taxi in gas location
        """
        taxi_name = refuel_action[1]
        i, j = state['taxis'][taxi_name]['location']
        if state['map'][i][j] == 'G':
            return True
        else:
            return False

    taxis = state["taxis"]
    passengers = state["passengers"]
    matrix = state["map"]
    rows = len(matrix)
    cols = len(matrix[0])
    acts = {}
    for taxi in taxis:
        acts[taxi] = []
        x, y = state["taxis"][taxi]["location"]
        fuel = state["taxis"][taxi]["fuel"]
        if 0 <= x + 1 < rows and 0 <= y < cols and matrix[x + 1][y] != IMPASSABLE and fuel > 0:
            acts[taxi].append(("move", taxi, (x+1, y)))
        if 0 <= x - 1 < rows and 0 <= y < cols and matrix[x - 1][y] != IMPASSABLE and fuel > 0:
            acts[taxi].append(("move", taxi, (x - 1, y)))
        if 0 <= x < rows and 0 <= y + 1 < cols and matrix[x][y + 1] != IMPASSABLE and fuel > 0:
            acts[taxi].append(("move", taxi, (x, y + 1)))
        if 0 <= x < rows and 0 <= y - 1 < cols and matrix[x][y - 1] != IMPASSABLE and fuel > 0:
            acts[taxi].append(("move", taxi, (x, y - 1)))

        acts[taxi] += [("pick up", taxi, passenger) for passenger in passengers
                       if _is_pick_up_action_legal(("pick up", taxi, passenger))]

        for passenger in passengers:
            if _is_pick_up_action_legal(("pick up", taxi, passenger)):
                acts[taxi].append(("pick up", taxi, passenger))
            if _is_drop_action_legal(("drop off", taxi, passenger)):
                acts[taxi].append(("drop off", taxi, passenger))

        if _is_refuel_action_legal(("refuel", taxi)):
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
        self.state["turns to go"] -= 1
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
    return _p(new_state)


def reward(state, action):
    r = 0
    if action == "reset":
        r += -50
    for atomic_action in action:
        act = atomic_action[0]
        if act == "drop off":
            r += 100
        elif act == "refuel":
            r += -10
    return r


def VI(state, action, t):
    policy = list(range(t))

    def value_iterations(state, action, t):
        if t == 0:
            return reward(state, action)

        max_val = 0
        opt_act = None
        for act in actions(state):
            temp = apply(state, act) # All the states and their probabilities
            print(temp[1])
            val = sum([l[1] * value_iterations(l[0], act, t-1) for l in temp])
            if val > max_val:
                max_val = val
                opt_act = act
        policy[t-1] = opt_act
        return reward(state, action) + max_val
    value_iterations(state, action, t)
    return policy


class OptimalTaxiAgent:
    def __init__(self, initial):
        self.initial = deepcopy(initial)
        VI(self.initial, None, 100)

    def act(self, state):
        raise NotImplemented


class TaxiAgent:
    def __init__(self, initial):
        self.initial = initial

    def act(self, state):
        raise NotImplemented
