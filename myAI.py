
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch as torch
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

class net(nn.Module):
    def __init__(self, num_inputs=4, num_actions=4):
        super(net, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 7)
        self.fc2 = nn.Linear(7, num_actions)

    def forward(self, x):
        x = F.relu(x)
        x = F.relu(x)
        return x





class myAI(object):
    nn = None
    database = None
    s1 = None
    s2 = None
    action = None
    def __init__(self, gateway):
        self.gateway = gateway
        self.nn = net()
        print(self.nn)
        self.database = []
        self.s1 = {"dist_x": None, "dist_y": None, "my_hp": None, "opp_hp": None}
        self.s2 = {"dist_x": None, "dist_y": None, "my_hp": None, "opp_hp": None}
    def get_reward(self, hp1, hp2):
        return -1*(hp1 - hp2)

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

        return 0

    def input(self):
        # Return the input for the current frame
        return self.inputKey

    def save_state(self, state, action, reward, prime_state):
        temp = {"s1": state, "action": action, "reward": reward, "s2": prime_state}
        self.database.add(temp)

    def act(self, action):
        if action == 0:
                return "A"

        elif action == 1:
            return "B"
        elif action == 2:
            return "FOR_JUMP"
        else:
          return "THROW_HIT"



    def processing(self):

        # Just compute the input for the current frame
        if self.frameData.getEmptyFlag() or self.frameData.getRemainingFramesNumber() <= 0:
                self.isGameJustStarted = True
                return

        if self.cc.getSkillFlag():
                self.inputKey = self.cc.getSkillKey()
                print("****skipping frame****")
                return

        if not self.cc.getSkillFlag() and self.s1["dist_x"] is not None:
        # if self.frameData.getCharacter(self.player).isControl() and self.s1["dist_x"] is not None:
            print("------------ ")
            print("action taken: ", self.action)
            my_r = self.s1["my_hp"] - self.frameData.getCharacter(self.player).getHp()
            opp_r = self.s1["opp_hp"] - self.frameData.getCharacter(not self.player).getHp()
            print("MY hp : ", self.s1["my_hp"] - self.frameData.getCharacter(self.player).getHp())
            print("OPP hp: ", self.s1["opp_hp"] - self.frameData.getCharacter(not self.player).getHp())
            self.s1["dist_x"] = None
            print("total reward: ", opp_r - my_r )

        # self.inputKey.empty()
        # self.cc.skillCancel()

        # Just spam kick
        # self.cc.commandCall("B")
        eps = np.random.rand()

        my_char = self.frameData.getCharacter(self.player)
        opp_char = self.frameData.getCharacter(not self.player)
        dist_x = self.frameData.getDistanceX()
        dist_y = self.frameData.getDistanceY()
        my_hp = my_char.getHp()
        opp_hp = opp_char.getHp()



        # print("taking action: ", action)
        if True:

            if eps >= 0.9:
                action = np.random.randint(0, 4)
                print("taking action--- RG: ", action)
            else:
                out = self.nn(torch.tensor([dist_x, dist_y, my_hp, opp_hp]))
                self.action = out.max(0)[1].item()
                print("taking action NN:  ", self.action)
            self.cc.commandCall(self.act(self.action))


        # if self.s1["dist_x"] is None:

            self.s1["dist_x"] = dist_x
            self.s1["dist_y"] = dist_y
            self.s1["my_hp"] = my_hp
            self.s1["opp_hp"] = opp_hp



    # This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]
