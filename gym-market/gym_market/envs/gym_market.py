import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging

MAX_ACCOUNT_BALANCE = 2147483647

INITIAL_ACCOUNT_BALANCE = 10000

MAX_STEPS = 3000

MAX_NUM_SHARES = 2147483647

MAX_SHARE_PRICE = 1000

logger = logging.getLogger(__name__)


class Market(gym.Env):
    metadata = {'render.modes': ['human']}

    def _get_info(self):
        return {}

    def __init__(self, lob_data, tape_data):
        super(Market, self).__init__()
        self.lob_data = lob_data
        self.tape_data = tape_data
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        # 0:buy,1:sell,2:hold
        # (0 - 1) * 10 seconds
        # self.action_space = spaces.Tuple((spaces.Box(low=0, high=3, shape=(1,), dtype=np.float32),
        #                                  spaces.Box(low=0, high=3, shape=(1,), dtype=np.float32),
        #                                  spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        #                                  spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)))
        # self.action_space = spaces.Tuple((spaces.Box(low=-1.5, high=1.5, shape=(1,), dtype=np.float64),
        #                                   spaces.Box(low=-1.5, high=1.5, shape=(1,), dtype=np.float64),
        #                                   spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float64),
        #                                   spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float64)))
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(201,), dtype=np.float64)
        self.action_history = np.zeros(12)

    def _next_observation(self):
        zeros_array = np.zeros(183)  # assume there are 61 features in lob_data
        frame = np.array(self.lob_data.iloc[self.current_step])

        if self.current_step == 0:
            zeros_array[-len(frame):] = frame
        if self.current_step == 1:
            frame = np.concatenate((frame, np.array(self.lob_data.iloc[self.current_step - 1])))
            zeros_array[-len(frame):] = frame
        if self.current_step != 0 and self.current_step != 1:
            frame = np.concatenate((frame, np.array(self.lob_data.iloc[self.current_step - 1])))
            frame = np.concatenate((frame, np.array(self.lob_data.iloc[self.current_step - 2])))
            zeros_array[-len(frame):] = frame

        obs = np.append(zeros_array, [  # append length = 6
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ])
        obs = np.append(obs, self.action_history)  # append length = 12
        return obs

    def _take_one_action(self, action):
        action_true_movement_time = self.current_step * 10 + ((action[1] + 1) / 2) * 10
        self.tape_data['time_differ'] = abs(self.tape_data['Time'] - action_true_movement_time)
        nearest_index = self.tape_data['time_differ'].idxmin()
        current_price = self.tape_data.loc[nearest_index]['Weighted_Price']

        action_type = (action[0] + 1) * 1.5
        amount = 1

        if action_type < 1:
            # buy
            shares_bought = amount
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            if self.shares_held + shares_bought != 0:
                self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            else:
                self.cost_basis = 0
            self.shares_held += shares_bought

        elif action_type < 2:
            # sell
            shares_sold = amount
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def _take_action(self, actions):
        self._take_one_action(np.array([actions[0], actions[2]]))
        self._take_one_action(np.array([actions[1], actions[3]]))
        self.action_history = np.append(self.action_history, actions)
        self.action_history = np.delete(self.action_history, [0, 1, 2, 3])

    def step(self, action):
        self._take_action(action)

        self.current_step += 1

        if self.current_step >= len(self.lob_data):
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier

        terminated = self.current_step >= len(self.lob_data) - 1

        truncated = bool(self.net_worth <= 0)

        obs = self._next_observation()

        return obs, reward, terminated ,truncated,self._get_info()

    def reset(self, seed=None, options=None):
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        self.current_step = 0

        return self._next_observation(),self._get_info()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
