# dependencies
import numpy as np
import time

class qlfa_agent(object):

    s1 = None
    s2 = None
    actions_map = None
    gamma = 0.99
    num_actions = 9
    t = None
    reward_scale = 30

    def __init__(self, gateway):
        self.gateway = gateway
        self.database = []
        self.s1 = None
        self.s2 = None
        self.t = 0
        self.epsilon = 0.1
        self.action = 1
        self.action_prev = 1
        # setting up actions dictionary

        self.actions_map = ["DASH", "STAND_GUARD", "FOR_JUMP", "STAND_D_DF_FA", "STAND_D_DF_FB", "A", "B", "THROW_A",
                    "AIR_DB", "AIR_DA"]
        self.num_actions = len(self.actions_map)
        self.weights = np.load("trained_weights/lfa_weights/model181009-204540.npy")
        print("--- Successfully Loaded Model ---")

    def close(self):
        pass

    def getInformation(self, frameData):
        # Getting the frame data of the current frame
        self.frameData = frameData
        self.cc.setFrameData(self.frameData, self.player)

    def roundEnd(self, x, y, z):
        print(x)

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
        sw = self.gameData.getStageWidth()
        sh = self.gameData.getStageHeight()

        my_char = self.frameData.getCharacter(self.player)
        opp_char = self.frameData.getCharacter(not self.player)
        dist_x = self.frameData.getDistanceX()/sw
        dist_y = self.frameData.getDistanceY()/sh
        my_state = self.decode_state(my_char.getState())
        opp_state = self.decode_state(opp_char.getState())

        my_energy = my_char.getEnergy()/300
        opp_energy = opp_char.getEnergy()/300
        my_spdx = my_char.getSpeedX()/15
        my_spdy = my_char.getSpeedY()/28
        opp_spdx = opp_char.getSpeedX()/15
        opp_spdy = opp_char.getSpeedY()/28
        my_hp = my_char.getHp()/500
        opp_hp = opp_char.getHp()/500
        diff_hp = (my_hp - opp_hp)/50
        my_center_x = my_char.getCenterX()/sw
        my_center_y = my_char.getCenterY()/sh
        opp_center_x = opp_char.getCenterX()/sw
        opp_center_y = opp_char.getCenterY()/sh
        my_char_hit_x = my_char.getRight()/sw
        my_char_hit_y = my_char.getLeft()/sw
        opp_char_hit_x = opp_char.getRight()/sw
        opp_char_hit_y = opp_char.getLeft()/sw
        my_char_hc = my_char.getHitCount()/9
        opp_char_hc = opp_char.getHitCount()/9
        dist_wall = (my_center_x - self.gameData.getStageWidth())/sw
        opp_act = opp_char.getAction().ordinal()/41
        my_act = my_char.getAction().ordinal()/41

        oppProjectiles = self.frameData.getProjectilesByP2()
        myProjectiles = self.frameData.getProjectilesByP1()

        if len(oppProjectiles) == 1:
            opp_proj_d1 = oppProjectiles[0].getHitDamage()/50
            opp_proj_x1 = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[0].getCurrentHitArea().getRight())/2)/960.0
            opp_proj_y1 = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[0].getCurrentHitArea().getBottom())/2)/640.0
            opp_proj_d2 = 0.0
            opp_proj_x2 = 0.0
            opp_proj_y2 = 0.0

        elif len(oppProjectiles) == 2:
            opp_proj_d1 = oppProjectiles[0].getHitDamage()/50
            opp_proj_x1 = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[0].getCurrentHitArea().getRight())/2)/960.0
            opp_proj_y1 = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[0].getCurrentHitArea().getBottom())/2)/640.0
            opp_proj_d2 = oppProjectiles[1].getHitDamage()/50
            opp_proj_x2 = ((oppProjectiles[1].getCurrentHitArea().getLeft() + oppProjectiles[1].getCurrentHitArea().getRight())/2)/960.0
            opp_proj_y2 = ((oppProjectiles[1].getCurrentHitArea().getTop() + oppProjectiles[1].getCurrentHitArea().getBottom())/2)/640.0
        else:
            opp_proj_d1 = 0.0
            opp_proj_x1 = 0.0
            opp_proj_y1 = 0.0
            opp_proj_d2 = 0.0
            opp_proj_x2 = 0.0
            opp_proj_y2 = 0.0

        if len(myProjectiles) == 1:
            my_proj_d1 = myProjectiles[0].getHitDamage()/50
            my_proj_x1 = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[0].getCurrentHitArea().getRight())/2)/960.0
            my_proj_y1 = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[0].getCurrentHitArea().getBottom())/2)/640.0
            my_proj_d2 = 0.0
            my_proj_x2 = 0.0
            my_proj_y2 = 0.0

        elif len(myProjectiles) == 2:

            my_proj_d1 = myProjectiles[0].getHitDamage()/50
            my_proj_x1 = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[0].getCurrentHitArea().getRight())/2)/960.0
            my_proj_y1 = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[0].getCurrentHitArea().getBottom())/2)/640.0
            my_proj_d2 = myProjectiles[1].getHitDamage()/50
            my_proj_x2 = ((myProjectiles[1].getCurrentHitArea().getLeft() + myProjectiles[1].getCurrentHitArea().getRight())/2)/960.0
            my_proj_y2 = ((myProjectiles[1].getCurrentHitArea().getTop() + myProjectiles[1].getCurrentHitArea().getBottom())/2)/640.0
        else:
            my_proj_d1 = 0.0
            my_proj_x1 = 0.0
            my_proj_y1 = 0.0
            my_proj_d2 = 0.0
            my_proj_x2 = 0.0
            my_proj_y2 = 0.0

        s = np.array([1, dist_x, dist_y, my_hp, opp_hp, my_state, opp_state, my_energy, opp_energy,
                          my_spdx, my_spdy, opp_spdx, opp_spdy, diff_hp, my_center_x, my_center_y, dist_wall,
                          opp_center_x, opp_center_y, my_char_hit_x, my_char_hit_y, opp_char_hit_x,
                          opp_char_hit_y, my_char_hc, opp_char_hc, opp_proj_d1, opp_proj_x1, opp_proj_y1,
                          opp_proj_d2, opp_proj_x2, opp_proj_y2, my_proj_d1, my_proj_x1, my_proj_y1,
                          my_proj_d2, my_proj_x2, my_proj_y2, opp_act, my_act])

        return s

    def Q_value(self, s, a):

        w_a = self.weights[a, :]
        dot_prod = np.dot(w_a, s)

        return dot_prod


    def processing(self):

        done_action = 10

        # Just compute the input for the current frame
        if self.frameData.getEmptyFlag() or self.frameData.getRemainingFramesNumber() <= 0:
            self.isGameJustStarted = True
            return

        if self.cc.getSkillFlag():
            self.inputKey = self.cc.getSkillKey()
            return

        self.inputKey.empty()
        self.cc.skillCancel()
        self.s1 = self.state_data()
        Q_vals = np.array([self.Q_value(self.s1, a) for a in range(len(self.actions_map))])
        self.action = Q_vals.argmax()
        self.cc.commandCall(self.act(self.action))


# This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]
