from py4j.java_gateway import get_field


class SandBag(object):
    def __init__(self, gateway):
        self.gateway = gateway

    def roundEnd(self, x, y, z):
    	print(x)
    	print(y)
    	print(z)
    	
    # please define this method when you use FightingICE version 4.00 or later
    def getScreenData(self, sd):
    	pass
    
    def close(self):
        pass

    def getInformation(self, frameData):
        pass

    def initialize(self, gameData, player):
        return 0

    def input(self):
        pass

    def processing(self):
        pass

    # This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]
