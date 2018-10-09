import sys
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters, get_field

from machete import Machete
from MultiHead import MultiHead


def check_args(args):
    for i in range(argc):
        if args[i] == "-n" or args[i] == "--n" or args[i] == "--number":
            global GAME_NUM
            GAME_NUM = int(args[i + 1])


def start_game():
    manager.registerAI("MultiHead", MultiHead(gateway))
    manager.registerAI("Machete", Machete(gateway))
    print("Start game")

    game = manager.createGame("ZEN", "ZEN", "MultiHead", "Machete", GAME_NUM)
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
GAME_NUM = 1
gateway = JavaGateway(gateway_parameters=GatewayParameters(port=4242),
                      callback_server_parameters=CallbackServerParameters());
manager = gateway.entry_point

main_process()