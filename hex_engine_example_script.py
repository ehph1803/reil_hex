#make sure that the module is located somewhere where your Python system looks for packages
import sys

#importing the module
from fhtw_hex import hex_engine as engine
#this is how you will provide the agent you generate during the group project
from fhtw_hex import example as eg
import test_machine as test
from actor import Actor

#initializing a game object
game = engine.hexPosition()
print(game.get_action_space())

#play the game against a random player, human plays 'black'
game.human_vs_machine(human_player=-1, machine=test.machine)

#play the game against the example agent, human play 'white'
#game.human_vs_machine(human_player=1, machine=eg.machine)
