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


class learnedAI(object):
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
    num_updates = None

    def __init__(self, gateway):
        self.gateway = gateway
        self.Q = Net()
        self.Q.load_state_dict(torch.load("model180801-065642.tar"))
        print("~~ successfully loaded weights ~~")
        self.database = []
        self.s1 = None
        self.s2 = None
        self.target_update_freq = 4
        self.rewards_per_round = 0
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

    def processing(self):

        # Just compute the input for the current frame
        if self.frameData.getEmptyFlag() or self.frameData.getRemainingFramesNumber() <= 0:
            self.isGameJustStarted = True
            return

        if self.cc.getSkillFlag():
            self.inputKey = self.cc.getSkillKey()
            self.count += 1
            return

        # elif not self.cc.getSkillFlag() and self.s1 is not None and self.count == 16 :
        #     print("------------ ")
        #     self.s2 = self.state_data()
        #     print("a: ", self.action)
        #     my_r = self.s1[2] - self.s2[2]
        #     opp_r = self.s1[3] - self.s2[3]
        #
        #     if my_r != 0 and opp_r != 0:
        #         print("$$$$$$$$$$$$$$$$$ double hit $$$$$$$$$$$$$$$$")
        #     # adjust the reward a little
        #     r = (opp_r - my_r).item()
        #
        #
        #     # if r == 0 and self.action > 2:
        #     #     r = -
        #
        #     print("reward: ", r)
        #     # if r == 0:
        #     #     print("r: ", r)
        #     # elif r < 0:
        #     #     r = r / (-r)
        #     #     print("r: ", r)
        #     # else:
        #     #     self.rewards_per_round = r
        #     #     # r = r / r
        #     #     print("r:  ", r)
        #     #     self.rewards_per_round = r
        #
        #     self.save_state(self.s1, self.action, r, self.s2)
        #     self.s1 = None

        self.inputKey.empty()
        self.cc.skillCancel()

        eps = np.random.rand()

        # if self.count == 16:
        s = self.state_data()
        out = self.Q(s)
        self.action = out.max(0)[1].item()
        # print("a: ", self.action)
        self.cc.commandCall(self.act(self.action))






    # This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]