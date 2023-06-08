#make sure that the module is located somewhere where your Python system looks for packages
#note that python does not search directory trees, hence you must provide the mother-directory of the package


#importing the module
from fhtw_hex import hex_engine as engine

#initializing a game object
game = engine.hexPosition()

#this is how your agent can be imported
#'submission' is the (sub)package that you provide
#please use a better name that identifies your group
from fhtw_hex.submission.facade import agent

#make sure that the agent you have provided is such that the following three
#method-calls are error-free and as expected

#let your agent play against random
game.machine_vs_machine(machine1=agent, machine2=None)
input("Press Enter to continue...")
game.machine_vs_machine(machine1=None, machine2=agent)
input("Press Enter to continue...")

#let your agent play against itself
game.machine_vs_machine(machine1=agent, machine2=agent)