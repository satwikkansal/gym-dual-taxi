import sys
import itertools
from contextlib import closing
from io import StringIO
from typing import Tuple, List

from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

MAP = [
    "+-------+",
    "|R: | :G|",
    "| : | : |",
    "| | : : |",
    "|Y: |B: |",
    "+-------+",
]


class DualTaxiEnv(discrete.DiscreteEnv):
    """
    Modified from the original taxi-v3 environment provided by openai-gym.

    Description:
    There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue).
    When the episode starts, the taxi starts off at a random square and the passenger is at a random location.
    The taxis drive to the passenger's location, picks up the passenger, drives to the passenger's destination
    (another one of the four specified locations), and then drops off the passenger. Either of the taxis can pick
    up the passenger. Once the passenger is dropped off, the episode ends. Once a passenger is picked up by a taxi,
    the other taxi is idle in the sense it loses its chance to pick and drop the passenger.

    Observations:
    There are 6144 discrete states since there are
    - 16 possible positions for each taxi (4x4 grid), so total 256 combinations
    - 6 possible locations of the passenger (including the cases when the passenger is in either of the taxi)
    - 4 destination locations

    Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi
    - 5: in taxi 2
    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)

    Actions:
    There are 6 discrete actions for each taxi:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger

    So over in all, there are 6x6 i.e. 36 possible actions.

    Rewards:
    - There is a default per-step reward of -2 for each taxi.
    - If the passenger is onboarded in other taxi this default per-step reward becomes -1 for the taxi
    without the passenger.
    - Dropping off passenger at correct location has reward of +100
    - Executing "pickup" and "drop-off" actions illegally, has reward of -10.
    - Collision has reward of -15 for each taxi.
    - Successful pickup has 0 reward for a taxi

    The environment can be intialized in two ways,
    - Competitive; Each taxi gets its own reward as feedback from the environment. Useful if you have to train two
                   independent agents.
    - Cooperative; The reward is combined, so the environment gives a joint reward as feedback. Useful if you have
                   to train a single agent that controls both the taxis and maximise the overall reward.

    These ways of initialization are controlled by a boolan argument named `competitive` that can be passed to the
    environment while initializing.

    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi 1
    - redL empty taxi 1 or 2
    - green: full taxi (can be either 1 or 2)
    - other letters (R, G, Y and B): locations for passengers and destinations
    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def apply_action(
            self, taxi_loc: Tuple, pass_idx: int, dest_idx: int, action, taxi_id: int):
        row, col = taxi_loc
        (new_row, new_col), new_pass_idx = taxi_loc, pass_idx
        done = False
        reward = -2

        if (pass_idx >= 4) and ((4 + taxi_id) != pass_idx):
            # passenger already onboarded in someone else's car
            # so the overall reward should be less -ve
            reward = -1

        if action == 0:  # North
            new_row = min(row + 1, self.max_row)
        elif action == 1:  # South
            new_row = max(row - 1, 0)
        # If the path is open, then only pass
        elif action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
            new_col = min(col + 1, self.max_col)
        elif action == 3 and self.desc[1 + row, 2 * col] == b":":
            new_col = max(col - 1, 0)
        elif action == 4:  # pickup
            if (pass_idx < 4 and taxi_loc == self.locs[pass_idx]):
                new_pass_idx = 4 + taxi_id
                reward = 0
            else:  # passenger not at location
                reward = -10
        elif action == 5:  # dropoff
            if (taxi_loc == self.locs[dest_idx]) and (pass_idx == (4 + taxi_id)):
                new_pass_idx = dest_idx
                done = True
                reward = 100
            elif (taxi_loc in self.locs) and (pass_idx == (4 + taxi_id)):
                # passenger deboarded at wrong stop
                new_pass_idx = self.locs.index(taxi_loc)
            else:  # Illegal dropoff action
                reward = -10

        new_loc = new_row, new_col
        return new_loc, new_pass_idx, done, reward

    def __init__(self, competitive=False):
        self.desc = np.asarray(MAP, dtype='c')
        self.competitive = competitive

        # Location of pickup / dropoff points on 4x4 grid
        self.locs = locs = [(0, 0), (0, 3), (3, 0), (3, 2)]

        # (4 x 4) x (6) x (4) x (4 x 4)
        num_states = 6144
        num_rows = 4
        num_columns = 4
        self.max_row = num_rows - 1
        self.max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = 6 * 6  # x 6
        # The transition probability map
        P = {state: {action: []
                     for action in range(num_actions)} for state in range(num_states)}

        self.stash = {}

        count = 0
        # Go over all the possible states and populate transition table
        # (which contains information about the transition prob, next state, reward, and the state is terminal one)
        for row_1, col_1 in itertools.product(range(num_rows), range(num_columns)):
            for row_2, col_2 in itertools.product(range(num_rows), range(num_columns)):
                for pass_idx in range(len(locs) + 2):  # +2 for being inside each of the taxis
                    for dest_idx in range(len(locs)):
                        count += 1

                        # This returns a number b/w zero and state space size
                        loc_1 = (row_1, col_1)
                        loc_2 = (row_2, col_2)
                        state = self.encode(loc_1, loc_2, pass_idx, dest_idx)

                        # The passenger and destination can't overlap
                        # The car locations can't overlap
                        if ((pass_idx < 4) and (pass_idx != dest_idx)) and (loc_1 != loc_2):
                            # Not all state combinations are valid combinations
                            initial_state_distrib[state] += 1

                        for action in range(num_actions):
                            action_1, action_2 = self.decode_action(action)

                            new_loc_1, new_pass_idx_1, done_1, reward_1 = self.apply_action(
                                loc_1, pass_idx, dest_idx, action_1, taxi_id=0
                            )

                            new_loc_2, new_pass_idx_2, done_2, reward_2 = self.apply_action(
                                loc_2, pass_idx, dest_idx, action_2, taxi_id=1
                            )

                            if (new_loc_1 == new_loc_2):
                                # collision!
                                reward_1 = reward_2 = -15
                                new_loc_1, new_loc_2 = loc_1, loc_2
                                new_pass_idx = pass_idx
                            elif (loc_1 == loc_2):
                                # spawned at collision
                                # Technically this should not encountered. So if you detect -0.5 rewards, something's off
                                done_1 = True
                                reward_1 = -0.5
                                reward_2 = 0
                                new_pass_idx = pass_idx
                            else:
                                # Resolving pass_idx
                                new_pass_idx = None
                                # after both the actions, the passenger location is unchanged
                                if new_pass_idx_1 == new_pass_idx_2 == pass_idx:
                                    new_pass_idx = new_pass_idx_1
                                # Taxi 1 pickup and passenger onboarded
                                elif (action_1 == 4) and (new_pass_idx_1 == 4):
                                    new_pass_idx = new_pass_idx_1
                                # Taxi 2 pickup and passenger onboarded
                                elif (action_2 == 4) and (new_pass_idx_2 == 5):
                                    new_pass_idx = new_pass_idx_2
                                # Taxi 1 finished
                                elif done_1:
                                    new_pass_idx = new_pass_idx_1
                                # Taxi 2 finished
                                elif done_2:
                                    new_pass_idx = new_pass_idx_2
                                # passenger was onboarded in taxi 1 and it took dropoff action
                                elif (pass_idx == 4) and (action_1 == 5):
                                    new_pass_idx = new_pass_idx_1
                                # passenger was onboarded in taxi 2 and it took dropoff action
                                elif  (pass_idx == 5) and (action_2 == 5):
                                    new_pass_idx = new_pass_idx_2
                                else:
                                    raise Exception(f"unksnown state with pass_idx {loc_1} {loc_2} {pass_idx} {done_1} {done_2} {action_1}:{new_pass_idx_1}, {action_2}:{new_pass_idx_2}")

                            new_state = self.encode(
                                new_loc_1, new_loc_2, new_pass_idx, dest_idx)

                            action = self.encode_action(action_1, action_2)

                            if self.competitive:
                                reward = (reward_1, reward_2)
                            else:
                                reward = reward_1 + reward_2

                            done = done_1 or done_2

                            # 1.0 indicate it's a deterministic env
                            P[state][action].append(
                                (1.0, new_state, reward, done))

        initial_state_distrib /= initial_state_distrib.sum()
        print('Total encoded states are', count)
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)

    def encode(self, taxi_loc_1: Tuple, taxi_loc_2: Tuple, pass_loc, dest_idx):
        taxi_row_1, taxi_col_1 = taxi_loc_1
        taxi_row_2, taxi_col_2 = taxi_loc_2
        i = taxi_row_1  # Row 1 becomes the most significant digit
        i *= 4  # This 5 is for col_1
        i += taxi_col_1
        i *= 4

        i += taxi_row_2
        i *= 4
        i += taxi_col_2
        i *= 6 # This 5 is for pass_loc

        i += pass_loc
        i *= 4
        i += dest_idx # dest_idx is the least significant digit

        return i

    def decode(self, i):
        out = []
        # In reverse order of encoding
        out.append(i % 4) # least significant digit i.e. destination first
        i = i // 4
        out.append(i % 6)  # Now the pass_loc
        i = i // 6

        col_2 = i % 4  # Now loc_2
        i = i // 4
        row_2 = i % 4
        i = i // 4
        out.append((row_2, col_2))

        col_1 = i % 4  # Now, finally loc_1
        i = i // 4
        row_1 = i
        out.append((row_1, col_1))
        assert 0 <= i < 4

        return reversed(out)

    def encode_action(self, action_1, action_2):
        return action_1 * 6 + action_2

    def decode_action(self, action):
        action_2 = action % 6
        action = action // 6
        action_1 = action % 6
        assert 0 <= action_1 < 6
        return (action_1, action_2)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxi_loc_1, taxi_loc_2, pass_idx, dest_idx = self.decode(self.s)

        def ul(x): return "_" if x == " " else x
        if pass_idx < 4:  # passenger not in taxi
            # Yellow color the first taxi
            taxi_row, taxi_col = taxi_loc_1
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], 'yellow', highlight=True)

            # Red color the second taxi
            taxi_row, taxi_col = taxi_loc_2
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], 'red', highlight=True)

            # Blue color the passenger location
            pi, pj = self.locs[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)
        else:  # passenger in taxi
            # Green color the taxi containing the passenger

            if pass_idx == 4:
                taxi_row, taxi_col = taxi_loc_1
                out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                    ul(out[1 + taxi_row][2 * taxi_col + 1]), 'green', highlight=True)

                taxi_row, taxi_col = taxi_loc_2
                out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                    ul(out[1 + taxi_row][2 * taxi_col + 1]), 'red', highlight=True)
            elif pass_idx == 5:
                taxi_row, taxi_col = taxi_loc_1
                out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                    ul(out[1 + taxi_row][2 * taxi_col + 1]), 'yellow', highlight=True)

                taxi_row, taxi_col = taxi_loc_2
                out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                    ul(out[1 + taxi_row][2 * taxi_col + 1]), 'green', highlight=True)
            else:
                raise Exception("Unknown state")

        di, dj = self.locs[dest_idx]
        # Magenta color the destination
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            a1, a2 = self.decode_action(self.lastaction)
            # TODO: figure out what this outfile is used for
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][a1]))
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][a2]))
        else:
            outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
