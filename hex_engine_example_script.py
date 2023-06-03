#make sure that the module is located somewhere where your Python system looks for packages
import sys

#importing the module
from fhtw_hex import hex_engine as engine
#this is how you will provide the agent you generate during the group project

import test_machine
import test_machine_random

#initializing a game object
game = engine.hexPosition()

#play the trained actor against a random machine
game.machine_vs_machine(machine1=test_machine.machine, machine2=test_machine_random.machine)
