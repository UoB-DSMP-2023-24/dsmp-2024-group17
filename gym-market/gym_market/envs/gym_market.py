import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

MAX_ACCOUNT_BALANCE = 30000

INITIAL_ACCOUNT_BALANCE = 10000

MAX_STEPS = 3000

MAX_NUM_SHARES = 100

MAX_SHARE_PRICE = 1000

logger = logging.getLogger(__name__)


class Market(gym.Env):
    metadata = {'render_modes': ['human']}

    def _get_info(self):
        return {}

    def _read_data(self, lob_data_dir, tape_data_dir):
        lob_data = pd.read_csv(lob_data_dir).dropna()
        self.tape_data = pd.read_csv(tape_data_dir)

        selected_features = ['start_time', 'log_return1_realized_volatility', 'log_return1_realized_volatility.1',
                             'log_return1_realized_volatility.2',
                             'log_return1_realized_volatility.3',
                             'log_return1_realized_volatility.4',
                             'log_return1_realized_volatility.5', 'total_volume_sum', 'total_volume_sum.1',
                             'total_volume_sum.2',
                             'total_volume_sum.3', 'total_volume_sum.4', 'total_volume_sum.5',
                             'transaction_quantity_sum', 'transaction_quantity_sum.1',
                             'transaction_quantity_sum.2', 'transaction_quantity_sum.3',
                             'transaction_quantity_sum.4', 'transaction_quantity_sum.5', 'transaction_count_sum',
                             'transaction_count_sum.1',
                             'transaction_count_sum.2', 'transaction_count_sum.3',
                             'transaction_count_sum.4', 'transaction_count_sum.5', 'price_spread_sum',
                             'price_spread_sum.1', 'price_spread_sum.2',
                             'price_spread_sum.3', 'price_spread_sum.4', 'price_spread_sum.5', 'bid_spread_sum',
                             'bid_spread_sum.1', 'bid_spread_sum.2',
                             'bid_spread_sum.3', 'bid_spread_sum.4', 'bid_spread_sum.5', 'ask_spread_sum',
                             'ask_spread_sum.1', 'ask_spread_sum.2',
                             'ask_spread_sum.3', 'ask_spread_sum.4', 'ask_spread_sum.5', 'volume_imbalance_sum',
                             'volume_imbalance_sum.1',
                             'volume_imbalance_sum.2', 'volume_imbalance_sum.3',
                             'volume_imbalance_sum.4', 'volume_imbalance_sum.5', 'bid_ask_spread_sum',
                             'bid_ask_spread_sum.1', 'bid_ask_spread_sum.2',
                             'bid_ask_spread_sum.3', 'bid_ask_spread_sum.4', 'bid_ask_spread_sum.5', 'size_tau2',
                             'size_tau2.1', 'size_tau2.2', 'size_tau2.3', 'size_tau2.4',
                             'size_tau2.5']

        lob_data = lob_data[selected_features]
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(lob_data.values)
        self.lob_data = pd.DataFrame(normalized_data, columns=lob_data.columns)

    def __init__(self, lob_data_dir, tape_data_dir, render_mode='human'):
        super(Market, self).__init__()
        self.lob_data_dir = lob_data_dir
        self.tape_data_dir = tape_data_dir
        self._read_data(self.lob_data_dir, self.tape_data_dir)
        self.render_mode = render_mode
        self.reward_range = (-1, 1)
        # 0:buy,1:sell,2:hold
        # (0 - 1) * 10 seconds
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(207,), dtype=np.float64)
        self.action_history = np.zeros(18)
        self.trade_data = []
        self.last_balance = INITIAL_ACCOUNT_BALANCE
        # self.sell_reward = [0, 0]

    def _next_observation(self):
        zeros_array = np.zeros(183)  # there are 61 features in lob_data
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
        obs = np.append(obs, self.action_history)  # append length = 18
        return obs

    def _take_one_action(self, action, action_number):
        action_true_movement_time = self.current_step * 10 + ((action[1] + 1) / 2) * 10
        self.tape_data['time_differ'] = abs(self.tape_data['Time'] - action_true_movement_time)
        nearest_index = self.tape_data['time_differ'].idxmin()
        current_price = self.tape_data.loc[nearest_index]['Weighted_Price']

        action_type = (action[0] + 1) * 1.5
        amount = int((action[2] + 1) * 5)
        flag = 0

        # if action_number == 0:
        #     self.sell_reward = []
        if self.current_step == 2999 and action_number == 1:
            action_type = 2
            amount = self.shares_held

        if action_type <= 1 and self.balance >= amount * current_price:
            # buy
            flag = 1
            shares_bought = amount
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            if self.shares_held + shares_bought != 0:
                self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            else:
                self.cost_basis = 0
            self.shares_held += shares_bought

        elif action_type <= 2 and self.shares_held >= amount:
            # sell
            flag = 1
            shares_sold = amount
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

            # sell_reward = shares_sold * (current_price - self.cost_basis)
            # self.sell_reward[action_number] = sell_reward

        elif action_type <= 3:
            # hold
            flag = 1
        #     current_step_end_time = (self.current_step + 1) * 10
        #     self.tape_data['time_differ'] = abs(self.tape_data['Time'] - current_step_end_time)
        #     nearest_index = self.tape_data['time_differ'].idxmin()
        #     current_price = self.tape_data.loc[nearest_index]['Weighted_Price']

        self.balances_of_each_action[action_number] = self.balance
        trade_info = [int(action_type), amount, current_price, flag]

        self.trade_data.append(trade_info)

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def _take_action(self, actions):
        self.balances_of_each_action = [0, 0]
        # flag = 0
        # if self.current_step == 2999:
        #     flag = 1
        #     print("final")
        #     actions[1] = 0
        #     actions[5] = self.shares_held
        #
        #     self._take_one_action(np.array([actions[0], actions[2], actions[4]]), 0)
        #     self._take_one_action(np.array([actions[1], actions[3], actions[5]]), 1)


        if actions[2] < actions[3]:
            self._take_one_action(np.array([actions[0], actions[2], actions[4]]), 0)
            self._take_one_action(np.array([actions[1], actions[3], actions[5]]), 1)
        else:
            self._take_one_action(np.array([actions[1], actions[3], actions[5]]), 0)
            self._take_one_action(np.array([actions[0], actions[2], actions[4]]), 1)
        self.action_history = np.append(self.action_history, actions)
        self.action_history = np.delete(self.action_history, [0, 1, 2, 3, 4, 5])

    def step(self, action):
        self._take_action(action)

        self.current_step += 1

        # if self.current_step >= len(self.lob_data):
        #     self.current_step = 0

        # delay_modifier = (self.current_step / MAX_STEPS)
        # reward = self.balance * delay_modifier

        # reward = (self.balance - INITIAL_ACCOUNT_BALANCE / INITIAL_ACCOUNT_BALANCE) * (self.current_step / MAX_STEPS)

        # reward = self.balance - self.last_balance
        # self.last_balance = self.balance

        # reward = (self.balances_of_each_action[1] - self.balances_of_each_action[0]) + (self.balance - self.last_balance)
        # self.last_balance = self.balance

        # reward = self.balance/INITIAL_ACCOUNT_BALANCE
        delay_modifier = (self.current_step / MAX_STEPS)
        reward = ((self.balance - INITIAL_ACCOUNT_BALANCE) * delay_modifier + (
                    self.balances_of_each_action[1] - self.balances_of_each_action[0]) + (
                              self.balance - self.last_balance)) / (3 * INITIAL_ACCOUNT_BALANCE)

        self.reward = reward

        terminated = False
        if self.current_step == MAX_STEPS:
            terminated = True
        # terminated = self.current_step >= MAX_STEPS

        truncated = bool(self.net_worth <= 0)

        obs = []

        if terminated is False:
            obs = self._next_observation()

        return obs, reward, terminated, truncated, self._get_info()

    def reset(self, seed=None, options=None):
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        self.current_step = 0

        self.trade_data = []

        self.last_balance = INITIAL_ACCOUNT_BALANCE
        return self._next_observation(), self._get_info()

    def render(self, close=False):
        # Render the environment to the screen
        if self.render_mode == 'human' and self.current_step > 0:
            profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

            print(f'Step: {self.current_step}')
            print(f'Balance: {self.balance}')
            print(f'Action0:{self.trade_data[self.current_step - 1]}')
            print(f'Action1:{self.trade_data[self.current_step]}')
            print(f'Reward: {self.reward}')
            print(
                f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
            print(
                f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
            print(
                f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
            print(f'Profit: {profit}')
