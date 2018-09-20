import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import torch as torch
import torch.optim as optim
import time


class Net(nn.Module):
    def __init__(self, num_inputs=38, num_actions=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, num_actions)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class tank(object):

    def __init__(self, gateway):
        self.gateway = gateway
        self.Q = Net()
        self.Q.load_state_dict(torch.load("trained_weights/tank_weights/rand_model180915-194313.tar"))
        print("---- successfully loaded weights ----")
        self.actions_map = ["DASH", "STAND_GUARD", "FOR_JUMP", "STAND_D_DF_FA", "A", "B", "THROW_A", "CROUCH_B",
                            "AIR_DB", "THROW_B"]
        self.action = 1

    def close(self):
        pass

    def getInformation(self, frameData):
        # Getting the frame data of the current frame
        self.frameData = frameData
        self.cc.setFrameData(self.frameData, self.player)

    def roundEnd(self, x, y, z):
        print(x)
        print(y)
        print(z)

    # please define this method when you use FightingICE version 4.00 or later
    def getScreenData(self, sd):
        pass

    def initialize(self, gameData, player):
        # Initializing the command center, the simulator and some other things
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
        # All the attributes that the state comprises of

        my_char = self.frameData.getCharacter(self.player)
        opp_char = self.frameData.getCharacter(not self.player)
        dist_x = self.frameData.getDistanceX()
        dist_y = self.frameData.getDistanceY()
        my_state = self.decode_state(my_char.getState())
        opp_state = self.decode_state(opp_char.getState)
        my_energy = my_char.getEnergy() / 300
        opp_energy = opp_char.getEnergy() / 300
        my_spdx = my_char.getSpeedX() / 15
        my_spdy = my_char.getSpeedY() / 28
        opp_spdx = opp_char.getSpeedX() / 15
        opp_spdy = opp_char.getSpeedY() / 28
        my_hp = my_char.getHp() / 500
        opp_hp = opp_char.getHp() / 500
        diff_hp = my_hp - opp_hp
        my_center_x = my_char.getCenterX()
        my_center_y = my_char.getCenterY()
        opp_center_x = opp_char.getCenterX()
        opp_center_y = opp_char.getCenterY()
        my_char_hit_x = my_char.getRight()
        my_char_hit_y = my_char.getLeft()
        opp_char_hit_x = opp_char.getRight()
        opp_char_hit_y = opp_char.getLeft()
        my_char_hc = my_char.getHitCount()
        opp_char_hc = opp_char.getHitCount()
        dist_wall = my_center_x - self.gameData.getStageWidth()
        opp_act = opp_char.getAction().ordinal()
        my_act = my_char.getAction().ordinal()

        oppProjectiles = self.frameData.getProjectilesByP2()
        myProjectiles = self.frameData.getProjectilesByP1()

        if len(oppProjectiles) == 1:
            opp_proj_d1 = oppProjectiles[0].getHitDamage() / 50
            opp_proj_x1 = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            opp_proj_y1 = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            opp_proj_d2 = 0.0
            opp_proj_x2 = 0.0
            opp_proj_y2 = 0.0

        elif len(oppProjectiles) == 2:
            opp_proj_d1 = oppProjectiles[0].getHitDamage() / 50
            opp_proj_x1 = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            opp_proj_y1 = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            opp_proj_d2 = oppProjectiles[1].getHitDamage()
            opp_proj_x2 = ((oppProjectiles[1].getCurrentHitArea().getLeft() + oppProjectiles[
                1].getCurrentHitArea().getRight()) / 2) / 960.0
            opp_proj_y2 = ((oppProjectiles[1].getCurrentHitArea().getTop() + oppProjectiles[
                1].getCurrentHitArea().getBottom()) / 2) / 640.0
        else:
            opp_proj_d1 = 0.0
            opp_proj_x1 = 0.0
            opp_proj_y1 = 0.0
            opp_proj_d2 = 0.0
            opp_proj_x2 = 0.0
            opp_proj_y2 = 0.0

        if len(myProjectiles) == 1:
            my_proj_d1 = myProjectiles[0].getHitDamage() / 50
            my_proj_x1 = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            my_proj_y1 = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            my_proj_d2 = 0.0
            my_proj_x2 = 0.0
            my_proj_y2 = 0.0

        elif len(myProjectiles) == 2:

            my_proj_d1 = myProjectiles[0].getHitDamage() / 50
            my_proj_x1 = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            my_proj_y1 = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            my_proj_d2 = myProjectiles[1].getHitDamage()
            my_proj_x2 = ((myProjectiles[1].getCurrentHitArea().getLeft() + myProjectiles[
                1].getCurrentHitArea().getRight()) / 2) / 960.0
            my_proj_y2 = ((myProjectiles[1].getCurrentHitArea().getTop() + myProjectiles[
                1].getCurrentHitArea().getBottom()) / 2) / 640.0
        else:
            my_proj_d1 = 0.0
            my_proj_x1 = 0.0
            my_proj_y1 = 0.0
            my_proj_d2 = 0.0
            my_proj_x2 = 0.0
            my_proj_y2 = 0.0

        s = torch.tensor([dist_x, dist_y, my_hp, opp_hp, my_state, opp_state, my_energy, opp_energy,
                          my_spdx, my_spdy, opp_spdx, opp_spdy, diff_hp, my_center_x, my_center_y, dist_wall,
                          opp_center_x, opp_center_y, my_char_hit_x, my_char_hit_y, opp_char_hit_x,
                          opp_char_hit_y, my_char_hc, opp_char_hc, opp_proj_d1, opp_proj_x1, opp_proj_y1,
                          opp_proj_d2, opp_proj_x2, opp_proj_y2, my_proj_d1, my_proj_x1, my_proj_y1,
                          my_proj_d2, my_proj_x2, my_proj_y2, opp_act, my_act]).type(torch.float32)
        return s

    def processing(self):

        # Just compute the input for the current frame
        if self.frameData.getEmptyFlag() or self.frameData.getRemainingFramesNumber() <= 0:
            self.isGameJustStarted = True
            return

        if self.cc.getSkillFlag():
            self.inputKey = self.cc.getSkillKey()
            return

        self.inputKey.empty()
        self.cc.skillCancel()

        s = self.state_data()
        out = self.Q(s)
        self.action = out.max(0)[1].item()
        self.cc.commandCall(self.act(self.action))

    class Java:
        implements = ["aiinterface.AIInterface"]
