from py4j.java_gateway import get_field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from AI_Module import ActionMap
import os
import csv
import copy

class MultiHead(object):
	# hyper parameters
	INPUT_SIZE = 141 # input layer size
	HIDDEN_SIZE= 80  # NN hidden layer size
	OUTPUT_SIZE = 40  # output layer size

	FloatTensor = torch.FloatTensor
	LongTensor = torch.LongTensor
	ByteTensor = torch.ByteTensor
	Tensor = FloatTensor

	Character = "ZEN" # ZEN or GARNET

	def __init__(self, gateway):
		self.gateway = gateway

	def close(self):
		pass

	def getInformation(self, frameData):
		# Load the frame data every time getInformation gets called
		self.frameData = frameData
		self.cc.setFrameData(self.frameData, self.player)

		# please define this method when you use FightingICE version 3.20 or later
	def roundEnd(self, x, y, z):
		print(x)
		print(y)
		print(z)

	def loadModel(self):
		# body load
		self.model.load_state_dict(torch.load("./Weight/"+self.Character+"/body.pth"))
		# offence Head load
		self.offenceHead.load_state_dict(torch.load("./Weight/"+self.Character+"/offence_head.pth"))
		# defence Head load
		self.defenceHead.load_state_dict(torch.load("./Weight/"+self.Character+"/defence_head.pth"))

	# please define this method when you use FightingICE version 4.00 or later
	def getScreenData(self, sd):
		pass

	def initialize(self, gameData, player):
		# Initializng the command center, the simulator and some other things
		self.inputKey = self.gateway.jvm.struct.Key()
		self.frameData = self.gateway.jvm.struct.FrameData()
		self.cc = self.gateway.jvm.aiinterface.CommandCenter()
		self.player = player  # p1 -> True, p2 -> False
		self.gameData = gameData
		self.simulator = self.gameData.getSimulator()
		self.isGameJustStarted = True

		self.actionMap = ActionMap()

		class Network(nn.Module):
			def __init__(self, inputSize, hiddenSize, outputSize):
				nn.Module.__init__(self)
				self.l1 = nn.Linear(inputSize, hiddenSize)
				self.l2 = nn.Linear(hiddenSize, hiddenSize)

			def forward(self, x):
				x = F.relu(self.l1(x))
				x = F.relu(self.l2(x))
				return x

		class OffenceHead(nn.Module):
			def __init__(self, hiddenSize, outputSize):
				nn.Module.__init__(self)
				self.l3 = nn.Linear(hiddenSize, outputSize)
			def forward(self, x):
				x = self.l3(x)
				return x

		class DefenceHead(nn.Module):
			def __init__(self, hiddenSize, outputSize):
				nn.Module.__init__(self)
				self.l3 = nn.Linear(hiddenSize, outputSize)
			def forward(self, x):
				x = self.l3(x)
				return x

		# body
		self.model = Network(self.INPUT_SIZE, self.HIDDEN_SIZE, self.OUTPUT_SIZE)
		# offence Head
		self.offenceHead = OffenceHead(self.HIDDEN_SIZE, self.OUTPUT_SIZE)
		# defence Head
		self.defenceHead = DefenceHead(self.HIDDEN_SIZE, self.OUTPUT_SIZE)

		# model load
		self.loadModel()

		return 0

	def input(self):
		return self.inputKey

	def getObservation(self):
		my = self.frameData.getCharacter(self.player)
		opp = self.frameData.getCharacter(not self.player)

		myHp = abs(my.getHp()/500)
		myEnergy = my.getEnergy()/300
		myX = ((my.getLeft() + my.getRight())/2)/960
		myY = ((my.getBottom() + my.getTop())/2)/640
		mySpeedX = my.getSpeedX()/15
		mySpeedY = my.getSpeedY()/28
		myState = my.getAction().ordinal()

		oppHp = abs(opp.getHp()/500)
		oppEnergy = opp.getEnergy()/300
		oppX = ((opp.getLeft() + opp.getRight())/2)/960
		oppY = ((opp.getBottom() + opp.getTop())/2)/640
		oppSpeedX = opp.getSpeedX()/15
		oppSpeedY = opp.getSpeedY()/28
		oppState = opp.getAction().ordinal()
		oppRemainingFrame = opp.getRemainingFrame()/70

		observation = []
		observation.append(myHp)
		observation.append(myEnergy)
		observation.append(myX)
		observation.append(myY)
		if mySpeedX < 0:
			observation.append(0)
		else:
			observation.append(1)
		observation.append(abs(mySpeedX))
		if mySpeedY < 0:
			observation.append(0)
		else:
			observation.append(1)
		observation.append(abs(mySpeedY))
		for i in range(56):
			if i == myState:
				observation.append(1)
			else:
				observation.append(0)

		observation.append(oppHp)
		observation.append(oppEnergy)
		observation.append(oppX)
		observation.append(oppY)
		if oppSpeedX < 0:
			observation.append(0)
		else:
			observation.append(1)
		observation.append(abs(oppSpeedX))
		if oppSpeedY < 0:
			observation.append(0)
		else:
			observation.append(1)
		observation.append(abs(oppSpeedY))
		for i in range(56):
			if i == oppState:
				observation.append(1)
			else:
				observation.append(0)
		observation.append(oppRemainingFrame)


		myProjectiles = self.frameData.getProjectilesByP1()
		oppProjectiles = self.frameData.getProjectilesByP2()

		if len(myProjectiles) == 2:
			myHitDamage = myProjectiles[0].getHitDamage()/200.0
			myHitAreaNowX = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[0].getCurrentHitArea().getRight())/2)/960.0
			myHitAreaNowY = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[0].getCurrentHitArea().getBottom())/2)/640.0
			observation.append(myHitDamage)
			observation.append(myHitAreaNowX)
			observation.append(myHitAreaNowY)
			myHitDamage = myProjectiles[1].getHitDamage()/200.0
			myHitAreaNowX = ((myProjectiles[1].getCurrentHitArea().getLeft() + myProjectiles[1].getCurrentHitArea().getRight())/2)/960.0
			myHitAreaNowY = ((myProjectiles[1].getCurrentHitArea().getTop() + myProjectiles[1].getCurrentHitArea().getBottom())/2)/640.0
			observation.append(myHitDamage)
			observation.append(myHitAreaNowX)
			observation.append(myHitAreaNowY)
		elif len(myProjectiles) == 1:
			myHitDamage = myProjectiles[0].getHitDamage()/200.0
			myHitAreaNowX = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[0].getCurrentHitArea().getRight())/2)/960.0
			myHitAreaNowY = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[0].getCurrentHitArea().getBottom())/2)/640.0
			observation.append(myHitDamage)
			observation.append(myHitAreaNowX)
			observation.append(myHitAreaNowY)
			for t in range(3):
				observation.append(0.0)
		else:
			for t in range(6):
				observation.append(0.0)

		if len(oppProjectiles) == 2:
			oppHitDamage = oppProjectiles[0].getHitDamage()/200.0
			oppHitAreaNowX = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[0].getCurrentHitArea().getRight())/2)/960.0
			oppHitAreaNowY = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[0].getCurrentHitArea().getBottom())/2)/640.0
			observation.append(oppHitDamage)
			observation.append(oppHitAreaNowX)
			observation.append(oppHitAreaNowY)
			oppHitDamage = oppProjectiles[1].getHitDamage()/200.0
			oppHitAreaNowX = ((oppProjectiles[1].getCurrentHitArea().getLeft() + oppProjectiles[1].getCurrentHitArea().getRight())/2)/960.0
			oppHitAreaNowY = ((oppProjectiles[1].getCurrentHitArea().getTop() + oppProjectiles[1].getCurrentHitArea().getBottom())/2)/640.0
			observation.append(oppHitDamage)
			observation.append(oppHitAreaNowX)
			observation.append(oppHitAreaNowY)
		elif len(oppProjectiles) == 1:
			oppHitDamage = oppProjectiles[0].getHitDamage()/200.0
			oppHitAreaNowX = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[0].getCurrentHitArea().getRight())/2)/960.0
			oppHitAreaNowY = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[0].getCurrentHitArea().getBottom())/2)/640.0
			observation.append(oppHitDamage)
			observation.append(oppHitAreaNowX)
			observation.append(oppHitAreaNowY)
			for t in range(3):
				observation.append(0.0)
		else:
			for t in range(6):
				observation.append(0.0)


		# print(len(observation))  #141
		#type(observation) -> list
		return observation

	def selectAction(self, state):
		# print("multi select max q value-----------------------")
		offenceQvalues = self.offenceHead(self.model(Variable(state, volatile=True).type(self.FloatTensor))).data
		defenceQvalues = self.defenceHead(self.model(Variable(state, volatile=True).type(self.FloatTensor))).data
		max_totalQ_index = (offenceQvalues + defenceQvalues).max(1)[1].view(1,1)

		return max_totalQ_index

	def playAction(self, actionNum):
		actionName = self.actionMap.actionMap[int(actionNum)]
		self.cc.commandCall(actionName)

	def processing(self):
		# First we check whether we are at the end of the round
		if self.frameData.getEmptyFlag() or self.frameData.getRemainingFramesNumber() <= 0:
			self.isGameJustStarted = True
			return
		if not self.isGameJustStarted:
			pass
		else:
			# If the game just started, no point on simulating
			self.isGameJustStarted = False

		if self.cc.getSkillFlag():
			self.inputKey = self.cc.getSkillKey()
			return
		self.inputKey.empty()
		self.cc.skillCancel()

		# DQN code
		state = self.getObservation()
		action = self.selectAction(self.FloatTensor([state]))
		self.playAction(action[0, 0])


	class Java:
		implements = ["aiinterface.AIInterface"]
