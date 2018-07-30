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

"""


class Net(nn.Module):
    def __init__(self, num_inputs=12, num_actions=7):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 12)
        self.fc2 = nn.Linear(12, num_actions)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        return x


class myAI(object):
    nn = None
    nn_prime = None
    database = None
    s1 = None
    s2 = None
    action = None
    count = 16
    actions_map = None
    lock = False
    optimizer = None
    target_update_freq = None
    rewards_per_round = None
    loss_fn = None

    def __init__(self, gateway):
        self.gateway = gateway
        self.Q = Net()
        self.Q.load_state_dict(torch.load("default_model.tar"))
        self.Q_target = Net()
        self.Q.load_state_dict(torch.load("default_model.tar"))
        print("~~ successfully loaded weights ~~")
        self.database = []
        self.s1 = None
        self.s2 = None
        self.target_update_freq = 4
        self.rewards_per_round = 0
        self.loss_fn = torch.nn.MSELoss()

        # setting up actions dictionary

        self.actions_map = ["A", "B", "FOR_JUMP _B B B", "AIR_DB", "FOR_JUMP", "AIR_B", "DASH"]
        # self.actions_map = ["A", "B", "FOR_JUMP _B B B", "AIR_DB", "BACK_JUMP", "STAND_D_DB_BB",
        #                     "STAND_F_D_DFA", "STAND_GUARD"]

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
        self.rewards_per_round = 0

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
        print(s)
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

        # Just compute the input for the current frame
        if self.frameData.getEmptyFlag() or self.frameData.getRemainingFramesNumber() <= 0:
            self.isGameJustStarted = True
            return

        if self.cc.getSkillFlag():
            self.inputKey = self.cc.getSkillKey()
            self.count += 1
            return

        elif not self.cc.getSkillFlag() and self.s1 is not None and self.count == 16:
            print("------------ ")
            self.s2 = self.state_data()
            print("a: ", self.action)
            my_r = self.s1[2] - self.s2[2]
            opp_r = self.s1[3] - self.s2[3]

            if my_r != 0 and opp_r != 0:
                print("$$$$$$$$$$$$$$$$$ double hit $$$$$$$$$$$$$$$$")
            # adjust the reward a little
            r = (opp_r - my_r).item()
            print("reward: ", r)
            # if r == 0:
            #     print("r: ", r)
            # elif r < 0:
            #     r = r / (-r)
            #     print("r: ", r)
            # else:
            #     self.rewards_per_round = r
            #     # r = r / r
            #     print("r:  ", r)
            #     self.rewards_per_round = r

            self.save_state(self.s1, self.action, r, self.s2)
            self.s1 = None

        self.inputKey.empty()
        self.cc.skillCancel()

        eps = np.random.rand()

        if self.count == 16:
            # with probability of 0.1 take a random action
            # else select with highest Q value
            if eps >= 0.9:
                self.action = np.random.randint(0, 7)
                # print("a: ", self.action)
            else:
                s = self.state_data()
                out = self.Q(s)
                self.action = out.max(0)[1].item()
                # print("a: ", self.action)
                self.cc.commandCall(self.act(self.action))

            self.s1 = self.state_data()
            self.count = 0
        else:
            self.count += 1

        """
        TO DO:
        
        write full DQN algorithm:
        
        1. check if the replay buffer is full.
        2. randomly sample from the replay buffer
        3. 
        
        """

        num_updates = 0

        if len(self.database) == 100 and not self.lock:

            self.lock = True
            print("%%%%%%%%%%%%%%%%%%%%%%%%%DB IS FULL%%%%%%%%%%%%%%%%%%%%%%")
            print("%%%%%%%%%%%%%% BEGIN TRAINING %%%%%%%%%%%%%%%%%%")

            batch_indices = np.random.randint(0, 100, 100)
            dbCopy = self.database.copy()
            self.database.clear()
            obs_batch = self.fetch(dbCopy, batch_indices, 0).type(torch.float32)
            rew_batch = self.fetch(dbCopy, batch_indices, 2).type(torch.float32)
            next_obs_batch = self.fetch(dbCopy, batch_indices, 3).type(torch.float32)

            print(self.Q(obs_batch)[1])

            current_Q_values = self.Q(obs_batch).max(1)[0].reshape(100, 1)

            # calculate the target's current Q values
            next_Q_values = self.Q_target(next_obs_batch).detach()
            next_Q_values = next_Q_values.max(1)[0].type(torch.float32)

            # find the target of the current Q values
            print("calculating target Q-values...")
            target_Q_values = rew_batch + 0.9*next_Q_values.reshape(100, 1)

            print("zeroing grads and calculating loss...")
            self.optimizer.zero_grad()
            loss = self.loss_fn(current_Q_values, target_Q_values)
            #
            print("backpropagating loss...")
            loss.backward()
            print("updating weights...")
            # perform parameter updates
            self.optimizer.step()
            num_updates += 1

            print(self.Q(obs_batch)[1])



            # update the target Q function
            if num_updates % self.target_update_freq == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())
                timestr = 'model' + time.strftime("%y%m%d-%H%M%S") + '.tar'
                torch.save(self.Q.state_dict(), timestr)
                print("successful save of model")

            self.lock = False

    # This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]
