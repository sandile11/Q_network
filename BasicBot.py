# -*- coding: utf-8 -*-

import sys
#set path
ai_data_path = '../data/aiData/BasicBot'
sys.path.append(ai_data_path)
import argparse
import importlib
import logging
import os
import re
import sys
import time
import traceback
import functools
import numpy as np
from py4j.java_gateway import get_field
from py4j.java_gateway import set_field
try:
    from tqdm import trange
except ImportError as exc:
    def _trange(n, *args, **kwargs):
        logger.info('Install "tqdm" to see progress bar. '
                    'Type "pip install tqdm" in terminal')
        return range(n)
    trange = _trange

from Config import *
import tfdeploy as td

logger = logging.getLogger('BasicBot')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
# logger can printout message from BasicBot(AIInterface)
# Because, Fighting Game Platform block print's functionality,
# We can't see error message, also can't see print message for debugging
# logging module is good solution for this case

if __name__ == '__main__':
    from Coach import Coach
    from FTGEnv import FTGEnv
    from Utils import *
    from Monitor import *
else:
    model_paths = ['BasicBot.pkl',
                   'BB/BasicBot.pkl']
    model, s1, q = None, None, None

    for model_path in model_paths:
        if os.path.exists(model_path):
            model = td.Model(model_path)
            s1, q = model.get('State', 'fully_connected_1/BiasAdd:0')
            print("Load weight file: {}".format(model_path))
            break

    if model is None:
        print("Can't find weight file")
        exit(1)


    def function_get_q_values(_state):
            return q.eval({state: _state})

    def act_with_np(state, stop_actions):
        screens, energy = state
        screens = screens.reshape([1, resolution[0], resolution[1], resolution[2]])
        q_values = q.eval({s1: screens})
        for stop_action in stop_actions:
            q_values[0][stop_action] = -1.0
        q_values[0] += 0.01 * np.random.randn(len(actions))
        return np.argmax(q_values[0])


class BasicBot(object):
    def __init__(self, gateway):
        self.gateway = gateway
        self.frames = frames
        self.frame_count = 0

        # save short period of data to make 4 channel data
        self.capacity = resolution[2] * 5
        self.screens = list()
        self.actions = list()
        self.hps = list()
        self.energy = list()
        self.controllable = list()
        self.rewards = list()

        # FTGEnv object set this
        # use this to avoid change API of this bot
        on_train = getattr(self, '_on_train', False)

        # when start game, subscribe self to Coach object
        # and override act, and memorise
        if on_train:
            Coach().trainees.append(self)
            self.act = functools.partial(Coach().act, eps=1.0)
            self.memorize = Coach().memorize
        else:
            self.act = act_with_np  # always do best action, epsilon == -1
            self.memorize = lambda s1, a, s2, done, r, energy: None  # do nothing

    # please define this method when you use FightingICE version 3.20 or later
    def roundEnd(self, x, y, z):
    	print(x)
    	print(y)
    	print(z)

    # Please define this method when you use FightingICE version 4.00 or later
    def getScreenData(self, sd):
        self.screenData = sd

    def close(self):
        pass

    def initialize(self, gameData, player):
        self.inputKey = self.gateway.jvm.struct.Key()
        self.frameData = self.gateway.jvm.struct.FrameData()
        self.cc = self.gateway.jvm.aiinterface.CommandCenter()

        self.player = player
        self.gameData = gameData
        return 0

    def getInformation(self, frameData):
        self.frameData = frameData
        self.cc.setFrameData(self.frameData, self.player)

    def input(self):
        return self.inputKey

    def processing(self):
        try:
            # initialize when game start (first processing call?)
            if self.frameData.getEmptyFlag() or self.frameData.getRemainingFramesNumber() <= 0:
                self.isGameJustStarted = True
                del self.screens[:]
                del self.actions[:]
                del self.hps[:]
                del self.energy[:]
                del self.controllable[:]
                del self.rewards[:]
                self.frame_count = self.frames
                return

            # if self.cc.getSkillFlag():
            #     self.inputKey = self.cc.getSkillKey()
            #     return

            # update state
            displayBuffer = self.screenData.getDisplayByteBufferAsBytes(resolution[1], resolution[0], True)
            # rescaled to [-1, 1] and change data type np.float32
            screen = np.frombuffer(displayBuffer, dtype=np.int8).reshape(resolution[:2]) / 127.0
            screen = screen.astype(np.float32)

            # opponent always inverted
            if self.player:
                screen = -screen
            self.screens.append(screen)

            my_char = self.frameData.getCharacter(self.player)
            opp_char = self.frameData.getCharacter(not self.player)
            self.hps.append((my_char.getHp(), opp_char.getHp()))
            my_energy = my_char.getEnergy()
            self.energy.append(my_energy / energy_scale)  # energy scaled

            # set action
            action_continue = self.actions and self.actions[-1] is not None \
                              and self.frame_count < self.frames
            # self.controllable.append(get_field(my_char, 'control'))
            self.controllable.append(not self.cc.getSkillFlag())
            controllable = self.controllable[-1]

            if my_char.isFront():
                actions = left_actions
            else:
                actions = right_actions

            # if last action is set and running
            if action_continue or not controllable:
                # keep current action
                self.actions.append(self.actions[-1])
            else:
                # take new action
                self.cc.skillCancel()
                stop_actions = [a for a in range(len(actions)) if energy_cost[a] > my_energy]
                action_idx = self.act(self._get_recent_state(), stop_actions=stop_actions)
                self.actions.append(action_idx)
                self.frames = len(actions[action_idx])
                logger.debug('action_idx: {}'.format(action_idx))
                self.frame_count = 0

            # set key
            if self.actions and self.actions[-1] is not None:
                current_action = actions[self.actions[-1]]
                if self.frame_count < len(current_action):
                    self.inputKey.empty()
                    action_keys = current_action[self.frame_count]
                    for key in action_keys:
                        if key in 'ABCDLRU':
                            set_field(self.inputKey, key, True)
            else:
                self.inputKey.empty()

            if getattr(self, '_on_train', False):
                # store new state action tuple and stream to monitor
                self.save_transaction(actions)
            self.frame_count += 1

        except Exception as exc:
            logger.error(traceback.format_exc())

    def save_transaction(self, actions):
        # save state and action tuple using Coach's memorize
        try:
            if len(self.screens) > self.capacity:
                del self.screens[0]
                del self.actions[0]
                del self.hps[0]
                del self.energy[0]
                del self.rewards[0]

            if len(self.screens) == self.capacity:
                assert(len(self.screens) == len(self.actions)
                       == len(self.hps) == len(self.energy))
                s1 = np.stack(
                    [self.screens[0],  # 1st frame
                     self.screens[4],  # 2nd frame
                     self.screens[8],  # 3rd frame
                     self.screens[12]], axis=2)  # 4th frame
                s2 = np.stack(
                    [self.screens[4],  # 2nd frame
                     self.screens[8],  # 3rd frame
                     self.screens[12],  # 4th frame
                     self.screens[16]], axis=2)  # 5th frame
                a = self.actions[0]
                # reward calculation
                my_hp_1, opp_hp_1 = self.hps[0]
                my_hp_2, opp_hp_2 = self.hps[4]
                energy = self.energy[0]
                r = (opp_hp_1 - opp_hp_2) - (my_hp_1 - my_hp_2)
                if opp_hp_1 >= my_hp_1:
                    r -= 0.1
                self.rewards.append(r)

                if self.controllable[0]:
                    # memorize sample when character is controllable
                    # do not use game end in this platform
                    # done is always false
                    self.memorize(s1, a, s2, False, r, energy)

                # for calculate score
                Coach().add_reward(r)

                # if there are debug port (process queue for monitoring code)
                # send current state and action  through this
                if hasattr(self, 'debug_port'):
                    cnt = getattr(self, '_debug_port_cnt', 0)
                    if cnt == 0:
                        screens, energy = self._get_recent_state(axis=0)
                        info = dict(action=str(actions[self.actions[-1]]),
                                    energy=int(energy * energy_scale),
                                    reward=r)
                        self.debug_port.put((screens, info))
                    setattr(self, '_debug_port_cnt', (cnt + 1) % 4)

        except Exception as exc:
            logger.error(traceback.format_exc())

    def _get_recent_state(self, axis=2):
        if len(self.screens) == self.capacity:
            screen = np.stack([self.screens[-13],  # 1st frame
                               self.screens[-9],  # 2nd frame
                               self.screens[-5],  # 3rd frame
                               self.screens[-1]], axis=axis)  # 4st frame
        else:
            screen = np.stack([self.screens[-1],
                               self.screens[-1],
                               self.screens[-1],
                               self.screens[-1]], axis=axis)
        energy = self.energy[-1]
        return screen, energy

    # This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="BasicBot example for Visual based Fighting Game AI Platform")
    parser.add_argument('train_or_test', nargs='+', choices=['train', 'test'],
                        help='Select Traning mode or Testing mode')
    parser.add_argument('--max-epoch', default=None, type=int, help='Max epochs for training')
    parser.add_argument('-ts', '--training-step', type=int, default=10000, help='set how many frame use for training')
    parser.add_argument('-es', '--evaluation-sec', type=int, default=30,
                        help='set how many time(seconds) use for evaluation (testing) without learning')
    parser.add_argument('-o', '--opponent', default='SandBag',
                        help='''opponent name
  - Python bot: opponent class name and filename shoud be same and it placed in same directory
  - Java bot: should be add character type like 'MctsAi:LUD', however it is not test enough''')
    parser.add_argument('-n', '--n-cpu', default=1, type=int,
                        help='set how many games launch, it should lower than number of CPU cores, '
                             'also considering limits of other computing resources')
    parser.add_argument('-p', '--path', default='.',
                        help='create this directory if it is not exists '
                             'and write csv file as a log and store networks parameters'
                             '(not working on python 2')
    parser.add_argument('--port', default=6000, type=int, help='help')
    parser.add_argument('-r', '--render', default='single', choices=['none', 'single', 'all'], help='game server port')
    parser.add_argument('-v, --verbose', dest='verbose', action='store_true', help='show debug level message')
    parser.set_defaults(verbose=False)
    parser.add_argument('-m', '--monitor', default='none',
                        choices=['none', 'pygame', 'matplotlib'],
                        help='show current input screen data and reward value and out action, '
                             'two types supported pygame and matplotlib')
    parser.add_argument('--starts-with-energy',dest='starts_with_energy', action='store_true',
                        help='get max energy for each character when start a game')
    parser.set_defaults(starts_with_energy=False)
    args = parser.parse_args()

    logger.info('Launch {:d} environment(s)'.format(args.n_cpu))
    logger.info('{} vs. {}'.format(bot_name, args.opponent))

    opponent = args.opponent
    if re.search("^[a-z0-9]+:(ZEN|LUD|GARNET)$", args.opponent, re.I):
        # for Java bot: eg. 'MctsAi:LUD'
        logger.info("Set java opponent: {}".format(args.opponent))
    else:

        # for python bot: bot's class name and file name should same,
        # the file should be place where this script can import

        try:
            mod = importlib.import_module(args.opponent)
            opponent = getattr(mod, args.opponent)
            logger.info("Set python opponent: {}".format(args.opponent))
        except ImportError:
            logger.error("Can't import {}".format(args.opponent))
            sys.exit(-1)

    root_path = args.path
    logger.info('Set path: {}'.format(root_path))
    make_dir(root_path)
    step_per_training = args.training_step
    seconds_per_testing = args.evaluation_sec

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug('Verbose mode on')

    logger.info('** Start {}ING ** '.format(args.train_or_test[0].upper()))
    # Train start
    if 'train' in args.train_or_test:
        logger.debug('Open {} as logfile to write'.format(os.path.join(root_path, log_file)))
        write_to_csv = make_csv_logger(root_path, log_file)

    sparkline_gen = make_sparkline('Scores')
    ai_monitor = make_monitor(args.monitor, env_id=0, player_no=1)

    envs = list()
    logger.info('Args: {}'.format(args))
    if 'train' in args.train_or_test:
        for n in range(args.n_cpu):
            disable_window = not (args.render == 'on' or args.render == 'single' and n == 0)
            env = FTGEnv(n, BasicBot, opponent, port=args.port + n,
                         inverted_player=1,
                         disable_window=disable_window,
                         starts_with_energy=args.starts_with_energy,
                         train='train' in args.train_or_test, verbose=False)
            env.run(block=False, ai_monitor=ai_monitor)
            envs.append(env)

        while Coach().memory.size < batch_size:
            time.sleep(5)
            pass

        epoch = 0
        start_time = timer()
        highest_score = -1e-10

        while True:
            Coach().training(epsilon)
            q_values = np.zeros(step_per_training)
            losses = np.zeros(step_per_training)
            for step in trange(step_per_training, desc='Training {}'.format(epoch)):
                max_q, loss = Coach().learn()
                q_values[step] = max_q.mean()
                losses[step] = loss
            training_score = Coach().get_rewards_stat()

            Coach().testing()
            for step in trange(seconds_per_testing, desc='Testing {}'.format(epoch)):
                time.sleep(1)
            testing_score = Coach().get_rewards_stat()

            logger.info('Epoch: {} Testing: {:.3f} ± {:.3f} Training: {:.3f} ± {:.3f} Loss: {:.9f} Q: {:.3f} ε: {:.5f}'.format(
                epoch, testing_score.mean(), testing_score.std(), training_score.mean(), training_score.std(),
                losses.mean(), q_values.mean(), epsilon))
            logger.info('Epoch: {} {}'.format(epoch, sparkline_gen(testing_score.mean())))

            write_to_csv(epoch, start_time, testing_score, training_score, losses)

            if testing_score.mean() > highest_score:
                highest_score = testing_score.mean()
                Coach().save(model_file, hint=highest_score)
                logger.info('New highest score: {}'.format(highest_score))

            # reduce epsilon to minimum (0.1) until half of the max_epoch
            epsilon_delta = 0.05 if args.max_epoch is None else 1.0 / args.max_epoch * 2
            epsilon = max(0.1, epsilon - epsilon_delta)
            epoch += 1
            if args.max_epoch is not None and epoch > args.max_epoch:
                break

    if 'test' in args.train_or_test:
        disable_window = not (args.render == 'on' or args.render == 'single')
        env = FTGEnv(0, BasicBot, opponent, port=args.port,
                     inverted_player=2,
                     disable_window=disable_window,
                     starts_with_energy=args.starts_with_energy,
                     train='train' in args.train_or_test, verbose=False)
        env.run(block=True, ai_monitor=ai_monitor)
        envs.append(env)

    logger.info('** Stop {}ING ** '.format(args.train_or_test[0].upper()))
    logger.info('Press ctrl + c many times to exit')
