# dependencies
import numpy as np
import time

class lfa(object):

    s1 = None
    s2 = None
    action = None
    act_frames_counter = 16
    actions_map = None
    rewards_per_round = []
    rewards_this_round = 0
    max_reward = 0
    num_updates = None
    parameter_update_counter = 0
    episode_counter = 0
    learning = True
    # Hyper-parameters for learning

    gamma = 0.99

    learning_freq = 20
    learning_rate = 0.000001
    num_actions = 7
    t = None
    reward_scale = 30

    def __init__(self, gateway):
        self.gateway = gateway
        print("~~ successfully loaded weights ~~")
        self.database = []
        self.s1 = None
        self.s2 = None
        self.t = 0
        self.num_updates = 0
        # setting up actions dictionary

        self.actions_map = ["DASH", "STAND_GUARD", "BACK_STEP", "A", "B", "THROW_A" ,"FOR_JUMP"]

        self.weights = np.load('trained_weights/model180905-073410.npy')
        print("successfully loaded numpy weights")

    def close(self):
        pass

    def getInformation(self, frameData):
        # Getting the frame data of the current frame
        self.frameData = frameData
        self.cc.setFrameData(self.frameData, self.player)

    def roundEnd(self, x, y, z):

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
        dist_wall = center_x - self.gameData.getStageWidth()

        s = np.array([1, dist_x, dist_y, my_hp, opp_hp, my_state, opp_state, my_energy, opp_energy,
                          my_spdx, my_spdy, opp_spdx, opp_spdy, diff_hp, center_x, center_y, dist_wall])

        return s

    def get_reward(self):

        my_char = self.frameData.getCharacter(self.player)
        opp_char = self.frameData.getCharacter(not self.player)
        my_hp = my_char.getHp()
        opp_hp = opp_char.getHp()

        return my_hp - opp_hp

    def Q_value(self, s, a):

        w_a = self.weights[a, :]
        dot_prod = np.dot(w_a, s)

        return dot_prod

    def processing(self):

        done_action = 5

        # Just compute the input for the current frame
        if self.frameData.getEmptyFlag() or self.frameData.getRemainingFramesNumber() <= 0:
            self.isGameJustStarted = True
            return

        if self.cc.getSkillFlag():
            self.inputKey = self.cc.getSkillKey()
            self.act_frames_counter += 1
            return

        self.inputKey.empty()
        self.cc.skillCancel()

        eps = np.random.rand()

        if self.act_frames_counter >= done_action:
            self.s1 = self.state_data()
            Q_vals = np.array([self.Q_value(self.s1, a) for a in range(len(self.actions_map))])
            self.action = Q_vals.argmax()
            self.cc.commandCall(self.act(self.action))
            self.act_frames_counter = 0
        else:
            self.act_frames_counter += 1





# This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]
