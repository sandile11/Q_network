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
    5. Find a way to update the replay buffer from the back

####Sandile
    1. still to investigate the reward function
    2. found some hyperparameters not sure how they work out yet
    3. later...
    4. I have printed out some statistics, need to test it out though
    5. I haven't done this

#### TO DO:
    CHECK IF THE PRINTING OF THE STATISTICS IS CORRECT
"""


class Net(nn.Module):
    def __init__(self, num_inputs=12, num_actions=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, num_actions)

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

    # Hyper-parameters for learning

    batch_size = 64
    gamma = 0.99
    replay_buffer_size = 10000
    target_update_freq = 10000
    learning_freq = 4
    num_actions = 10
    t = 0

    def __init__(self, gateway):
        self.gateway = gateway
        self.Q = Net()
        self.Q.load_state_dict(torch.load("default-model180801-040602.csv"))
        self.Q_target = Net()
        self.Q.load_state_dict(torch.load("default-model180801-040602.csv"))
        print("~~ successfully loaded weights ~~")
        self.database = []
        self.s1 = None
        self.s2 = None

        self.loss_fn = torch.nn.MSELoss()
        self.num_updates = 0
        # setting up actions dictionary

        self.actions_map = ["6 6 6", "STAND_GUARD", "BACK_STEP", "A", "B", "4 _ B",
                            "AIR_DB", "CROUCH_FB", "STAND_FA", "STAND_D_DF_FC"]

        # setting up my optimizer
        self.optimizer = optim.SGD(self.Q.parameters(), lr=0.01, momentum=0.0)

    def close(self):
        pass

    def getInformation(self, frameData):
        # Getting the frame data of the current frame
        self.frameData = frameData
        self.cc.setFrameData(self.frameData, self.player)

    # please define this method when you use FightingICE version 3.20 or later
    def roundEnd(self, x, y, z):
        print(x)
        print(y)
        print(z)
        print("^^^^^^TOTAL REWARDS FOR ROUND^^^^^^")
        print(self.rewards_this_round)
        if self.rewards_this_round > self.max_reward:
            self.max_reward = self.rewards_this_round
        self.rewards_per_round.append(self.rewards_per_round)
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
        self.episode_counter = 0
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

        s = torch.tensor([dist_x, dist_y, my_hp, opp_hp, my_state, opp_state, my_energy, opp_energy,
                          my_spdx, my_spdy, opp_spdx, opp_spdy]).type(torch.float32)
        # print(s)
        return s

    def fetch(self, db, index_arr, sp_index):
        # could be faster
        data = None
        for i in range(len(index_arr)):
            if data is None:
                data = db[index_arr[i]][sp_index]
            else:
                tmp = db[index_arr[i]][sp_index]
                data = np.vstack((data, tmp))
        # now pop out all of the sampled data out of the database

        return torch.tensor(data)

    def processing(self):

        self.t += 1

        done_action = 16

        # Just compute the input for the current frame
        if self.frameData.getEmptyFlag() or self.frameData.getRemainingFramesNumber() <= 0:
            self.isGameJustStarted = True
            return

        if self.cc.getSkillFlag():
            self.inputKey = self.cc.getSkillKey()
            self.act_frames_counter += 1
            return

        elif not self.cc.getSkillFlag() and self.s1 is not None and self.act_frames_counter == done_action:
            print("------------ ")
            self.s2 = self.state_data()
            print("a: ", self.action)
            my_r = self.s1[2] - self.s2[2]
            opp_r = self.s1[3] - self.s2[3]

            if my_r != 0 and opp_r != 0:
                print("$$$$$$$$$$$$$$$$$ double hit $$$$$$$$$$$$$$$$")
            r = (opp_r - my_r).item()
            self.rewards_this_round += r
            print("reward: ", r)
            self.save_state(self.s1, self.action, r, self.s2)
            self.s1 = None
            self.parameter_update_counter += 1

        self.inputKey.empty()
        self.cc.skillCancel()

        eps = np.random.rand()

        if self.act_frames_counter == done_action:
            # with probability of 0.1 take a random action
            # else select with highest Q value
            if eps >= 0.9:
                self.action = np.random.randint(0, self.num_actions)
                self.cc.commandCall(self.act(self.action))
                # print("a: ", self.action)
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

        if len(self.database) == self.replay_buffer_size and self.t > 1000000\
                and self.parameter_update_counter % self.learning_freq == 0:

            print("%%%%%%%%%%%%%%%%%%%%%%%%%DB IS FULL%%%%%%%%%%%%%%%%%%%%%%")
            print("%%%%%%%%%%%%%% BEGIN TRAINING %%%%%%%%%%%%%%%%%%")

            batch_indices = np.random.randint(0, self.replay_buffer_size, self.batch_size)
            dbCopy = self.database.copy()
            self.database.clear()
            obs_batch = self.fetch(dbCopy, batch_indices, 0).type(torch.float32)
            rew_batch = self.fetch(dbCopy, batch_indices, 2).type(torch.float32)
            next_obs_batch = self.fetch(dbCopy, batch_indices, 3).type(torch.float32)
            print("break 1")
            # print(self.Q(obs_batch)[1])

            current_Q_values = self.Q(obs_batch).max(1)[0].reshape(self.batch_size, 1)
            print("break 2")
            # calculate the target's current Q values
            next_Q_values = self.Q_target(next_obs_batch).detach()
            next_Q_values = next_Q_values.max(1)[0].type(torch.float32)

            # find the target of the current Q values
            print("calculating target Q-values...")
            target_Q_values = rew_batch + 0.9*next_Q_values.reshape(self.batch_size, 1)
            print("break 3")
            print("zeroing grads and calculating loss...")
            self.optimizer.zero_grad()
            loss = self.loss_fn(current_Q_values, target_Q_values)
            print("backpropagating loss...")
            loss.backward()
            print("updating weights...")
            # perform parameter updates
            self.optimizer.step()
            self.num_updates += 1

            # timestr = 'default-model' + time.strftime("%y%m%d-%H%M%S") + '.csv'
            # torch.save(self.Q.state_dict(), timestr)
            # print("successful save of model")

            # update the target Q function
            if self.num_updates % self.target_update_freq == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())
                timestr = 'model' + time.strftime("%y%m%d-%H%M%S") + '.tar'
                torch.save(self.Q.state_dict(), timestr)
                print("successful save of model")

        if self.episode_counter % 20 == 0:
            print("Timestep: " + time.strftime("%y%m%d-%H%M%S"))
            print("mean reward (100 episodes) ", np.array(self.rewards_per_round).sum()/len(self.rewards_per_round) )
            print("best reward: ", self.max_reward)
            print("episodes %d" % len(self.rewards_per_round))
            self.rewards_per_round.clear()
    # This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]
