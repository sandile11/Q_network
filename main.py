import sys

from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters, get_field
from Machete import Machete
from dqn import dqn
from dqn_agent import dqn_agent
from BasicBot import BasicBot
from lfa_qlearning import lfa_qlearning
from lfa_agent import lfa_agent
from dummy import dummy
from randomai import randomai
from MultiHead import MultiHead
from tank_agent import tank
from scorpion import scorpion

def check_args(args):
    for i in range(argc):
        if args[i] == "-n" or args[i] == "--n" or args[i] == "--number":
            global GAME_NUM
            GAME_NUM = int(args[i + 1])


def start_game():
    # manager.registerAI("dummy", dummy(gateway))
    # manager.registerAI("dqn", dqn(gateway))
    # manager.registerAI("lfa_qn", lfa_qlearning(gateway))
    manager.registerAI("tank", tank(gateway))
    # manager.registerAI("lfa_agent", lfa_agent(gateway))
    # manager.registerAI("random", randomai(gateway))
    manager.registerAI("scorpion", scorpion(gateway))
    # manager.registerAI("dqn_agent", dqn_agent(gateway))
    manager.registerAI("Machete", Machete(gateway))
    # manager.registerAI("basicbot", BasicBot(gateway))
    # manager.registerAI("multihead", MultiHead(gateway))
    # manager.registerAI("DisplayInfo", MctsAi(gateway))
    print("Start game")

    game = manager.createGame("ZEN", "ZEN", "tank", "scorpion", GAME_NUM)
    # game = manager.createGame("ZEN", "ZEN", "dqn_agent", "random", GAME_NUM)
    # game = manager.createGame("ZEN", "ZEN", "dqn", "Machete", GAME_NUM)
    # game = manager.createGame("ZEN", "ZEN", "Machete", "multihead", GAME_NUM)
    # game = manager.createGame("ZEN", "ZEN", "dummy", "random", GAME_NUM)
    #
    manager.runGame(game)

    print("After game")
    sys.stdout.flush()


def close_gateway():
    gateway.close_callback_server()
    gateway.close()


def main_process():
    check_args(args)
    start_game()
    close_gateway()


args = sys.argv
argc = len(args)
GAME_NUM = 60000
gateway = JavaGateway(gateway_parameters=GatewayParameters(port=6500),  callback_server_parameters=CallbackServerParameters())
manager = gateway.entry_point

main_process()
