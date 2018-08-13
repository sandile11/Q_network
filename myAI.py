import copy

import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import torch as torch
import torch.optim as optim
import time

"""
##TO DO

####features for neural network:
    1. distances between players
    2. attack is opponent attacking
    3. is opponent attacking
    4. player state
    5. distance between hit boxes
    6. opponent state

####reward function
    The difference between the HP would work

####STEVE:
    1. look at the reward function
    2. try to find the right hyperparameters
    3. you will have to compare this to an MCTS agent found online
    4. print out the statistics of the model so you can monitor convergence

####Sandile
    1. still to investigate the reward function
    2. found some hyperparameters not sure how they work out yet
    3. later...
    4. I have printed out some statistics, need to test it out though


#### TO DO:
    CHECK IF THE PRINTING OF THE STATISTICS IS CORRECT
    
#### DONE
    reward function okay
    printed statistics
    found a way to update the replay buffer
"""


class Net(nn.Module):
    def __init__(self, num_inputs=15, num_actions=7):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_actions)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class myAI(object):
    nn = None
    nn_prime = None
    database = None
    s1 = None
    s2 = None
    action = None
    act_frames_counter = 16
    actions_map = None
    lock = False
    optimizer = None
    rewards_per_round = []
    rewards_this_round = 0
    max_reward = 0
    loss_fn = None
    num_updates = None
    parameter_update_counter = 0
    episode_counter = 0
    learning = True
    # Hyper-parameters for learning

    batch_size = 32
    gamma = 0.99
    replay_buffer_size = 50000
    target_update_freq = 7000
    learning_freq = 20
    learning_rate = 0.01
    num_actions = 7
    t = None
    t_start_learning = 30000
    reward_scale = 30

    def __init__(self, gateway):
        self.gateway = gateway
        self.Q = Net()
        self.Q_target = copy.deepcopy(self.Q)
        print("~~ successfully loaded weights ~~")
        self.database = []
        self.s1 = None
        self.s2 = None
        self.t = 0
        self.loss_fn = torch.nn.MSELoss()
        self.num_updates = 0
        # setting up actions dictionary

        self.actions_map = ["DASH", "STAND_GUARD", "BACK_STEP", "A", "B", "THROW_A", "FOR_JUMP"]

        # setting up my optimizer
        self.optimizer = optim.SGD(self.Q.parameters(), lr=self.learning_freq, momentum=0.0)

    def close(self):
        pass

    def getInformation(self, frameData):
        # Getting the frame data of the current frame
        self.frameData = frameData
        self.cc.setFrameData(self.frameData, self.player)

    # please define this method when you use FightingICE version 3.20 or later
    def roundEnd(self, x, y, z):
        # print(x)
        # print(y)
        # print(z)
        # print("_______TOTAL REWARDS FOR ROUND______")
        # print(self.rewards_this_round)

        print("episode no: ", self.episode_counter)

        if self.frameData.getRound() % 3 == 0:
            self.rewards_per_round.append(self.rewards_this_round)

        if self.episode_counter % 25 == 0 and self.frameData.getRound() % 3 == 0:
            print("*************************************")
            print("Timestep: " + time.strftime("%y%m%d-%H%M%S"))
            print("mean reward per game ", np.array(self.rewards_per_round) / (3))

    # please define this method when you use FightingICE version 4.00 or later
    def getScreenData(self, sd):
        pass

    def initialize(self, gameData, player):
        # Initializng the command center, the simulator and some other things
        self.inputKey = self.gateway.jvm.struct.Key()
        self.frameData = self.gateway.jvm.struct.FrameData()
        self.cc = self.gateway.jvm.aiinterface.CommandCenter()
        self.player = player
        self.gameData = gameData
        self.simulator = self.gameData.getSimulator()
        self.isGameJustStarted = True
        self.episode_counter += 1
        self.rewards_this_round = 0
        return 0

    def input(self):
        # Return the input for the current frame
        return self.inputKey

    def save_state(self, state, action, reward, prime_state):
        temp = [state, action, reward, prime_state]

        self.database.append(temp)

    def act(self, action):
        return self.actions_map[action]

    def decode_state(self, state):
        if state == "CROUCH":
            return 0
        elif state == "STAND":
            return 1
        elif state == "AIR":
            return 2
        else:
            return 3

    def state_data(self):
        # All the attributes that the state comprises of

        my_char = self.frameData.getCharacter(self.player)
        opp_char = self.frameData.getCharacter(not self.player)
        dist_x = self.frameData.getDistanceX()
        dist_y = self.frameData.getDistanceY()
        my_state = self.decode_state(my_char.getState())
        opp_state = self.decode_state(opp_char.getState)
        my_energy = my_char.getEnergy()
        opp_energy = opp_char.getEnergy()
        my_spdx = my_char.getSpeedX()
        my_spdy = my_char.getSpeedY()
        opp_spdx = opp_char.getSpeedX()
        opp_spdy = opp_char.getSpeedY()
        my_hp = my_char.getHp()
        opp_hp = opp_char.getHp()
        diff_hp = my_hp - opp_hp
        center_x = my_char.getCenterX()
        center_y = my_char.getCenterY()

        s = torch.tensor([dist_x, dist_y, my_hp, opp_hp, my_state, opp_state, my_energy, opp_energy,
                          my_spdx, my_spdy, opp_spdx, opp_spdy, diff_hp, center_x, center_y]).type(torch.float32)

        return s

    def fetch(self, db, index_arr, sp_index):
        # fetches data from the replay buffer
        # could be faster
        data = None
        for i in range(len(index_arr)):
            if data is None:
                data = db[index_arr[i]][sp_index]
            else:
                tmp = db[index_arr[i]][sp_index]
                data = np.vstack((data, tmp))

        return torch.tensor(data)

    def get_reward(self):

        my_char = self.frameData.getCharacter(self.player)
        opp_char = self.frameData.getCharacter(not self.player)
        my_hp = my_char.getHp()
        opp_hp = opp_char.getHp()

        return my_hp - opp_hp

    def processing(self):

        done_action = 16

        # Just compute the input for the current frame
        if self.frameData.getEmptyFlag() or self.frameData.getRemainingFramesNumber() <= 0:
            self.isGameJustStarted = True
            return

        if self.cc.getSkillFlag():
            self.inputKey = self.cc.getSkillKey()
            self.act_frames_counter += 1
            return

        elif not self.cc.getSkillFlag() and self.s1 is not None and self.act_frames_counter >= done_action:

            # and self.frameData.getCharacter(self.player).isControl():
            self.s2 = self.state_data()
            my_r = self.s1[2] - self.s2[2]
            opp_r = self.s1[3] - self.s2[3]
            r = (opp_r - my_r).item()
            self.rewards_this_round += r

            # A different reward function

            if r <= 75:
                if r != 0:
                    r = r / self.reward_scale

                if self.s2[2] < self.s2[3]:
                    r -= 0.1

                    self.save_state(self.s1, self.action, r, self.s2)
            self.parameter_update_counter += 1
            self.rewards_this_round += r
            self.t += 1
        self.s1 = None

        self.inputKey.empty()
        self.cc.skillCancel()

        eps = np.random.rand()

        if self.act_frames_counter >= done_action:
        # and self.frameData.getCharacter(self.player).isControl():
        # with probability of 0.1 take a random action
        # else select with highest Q value
            if self.t < self.t_start_learning:
                self.action = np.random.randint(0, self.num_actions)
                self.cc.commandCall(self.act(self.action))
            elif eps >= 0.8:
                self.action = np.random.randint(0, self.num_actions)
                self.cc.commandCall(self.act(self.action))
            else:
                s = self.state_data()
                out = self.Q(s)
                self.action = out.max(0)[1].item()
                self.cc.commandCall(self.act(self.action))

            self.s1 = self.state_data()
            self.act_frames_counter = 0
        else:
            self.act_frames_counter += 1

    # Starting learning if conditions allow

        if len(self.database) == self.replay_buffer_size and self.t > self.t_start_learning \
                and self.parameter_update_counter % self.learning_freq == 0:
            if self.learning:
                print("_______________LEARNING_____________")
                self.learning = False
            batch_indices = np.random.randint(0, self.replay_buffer_size, self.batch_size)
            dbCopy = self.database.copy()
            obs_batch = self.fetch(dbCopy, batch_indices, 0).type(torch.float32)
            rew_batch = self.fetch(dbCopy, batch_indices, 2).type(torch.float32)
            next_obs_batch = self.fetch(dbCopy, batch_indices, 3).type(torch.float32)
            current_Q_values = self.Q(obs_batch).max(1)[0].reshape(self.batch_size, 1)

            next_Q_values = self.Q_target(next_obs_batch).detach()
            next_Q_values = next_Q_values.max(1)[0].type(torch.float32)
            target_Q_values = rew_batch + self.gamma * next_Q_values.reshape(self.batch_size, 1)
            self.optimizer.zero_grad()
            loss = self.loss_fn(current_Q_values, target_Q_values)
            loss.backward()
            self.optimizer.step()
            self.num_updates += 1
            for _ in range(20):
                self.database.pop(0)
            # update the target Q function
            if self.num_updates % self.target_update_freq == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())
                timestr = 'trained_weights/' + 'model' + time.strftime("%y%m%d-%H%M%S") + '.tar'
                torch.save(self.Q.state_dict(), timestr)
                print("_______________ successful save of new target Q")

# This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]
