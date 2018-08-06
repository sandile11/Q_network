# -*- coding: utf-8 -*-

import sys
import os
import time


if sys.version_info[0] == 3:
    _string = str
    make_dir = lambda path: os.makedirs(path, exist_ok=True)
    timer = time.perf_counter
elif sys.version_info[0] == 2:
    import string as _string
    make_dir = lambda path: path  # do nothing
    timer = time.clock


def _left2right(left_action):
    d = _string.maketrans('LR', 'RL')
    right_action = list()
    for a in left_action:
        if type(a) == int:
            right_action.append(a)
        else:
            right_action.append(_string.translate(a, d))
    return right_action


FTG_PATH = '..'
resolution = (38, 57, 4)
# (channel, height, width),
# channel: frames[t-3, t-2, t-1, t-0]
memory_size = 400000
batch_size = 32
learning_rate = 0.00025
gamma = 0.99  # discount factor
epsilon = 1.0  # initial epsilon (exploration rate) for epsilon greedy algorithm
reward_scale = 30.0   # scaled_reward = reward / reward_scale
energy_scale = 300.0  # scaled_energy = energy / energy_scale
frames = 4

bot_name = 'BasicBot'
log_file = '{}.csv'.format(bot_name)
model_file = '{}.pkl'.format(bot_name)
root_path = '.'


actions = [
    (0, 'L', '', '', ''),  # defense
    (0, 'LA', '', '', ''),  # throw

    (0, 'L', '', 'L', '', '', ''),
    (0, 'R', '', 'R', '', '', ''),

    (0, 'B', '', 'B', '', 'B', ''),  # B
    (0, 'DB', '', 'DB', '', 'DB', ''),  # 2_B

    (2, 'D', 'DR', 'RA', '', '', ''),  # 2 3 6_A
    (30, 'D', 'DR', 'RB', '', '', ''),  # 2 3 6_B
    # (150, 'D', 'DR', 'RC', '', '', ''),  # 2 3 6_C

    (0, 'D', 'LD', 'LA', '', '', ''),   # 2 1 4_A
    (50, 'D', 'LD', 'LB', '', '', ''),   # 2 1 4_B
]

energy_cost = [action[0] for action in actions]
left_actions = [action[1:] for action in actions]
right_actions = [_left2right(action) for action in left_actions]
