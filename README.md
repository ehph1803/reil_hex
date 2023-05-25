# reil_hex

## Implementation of HEX agent with A2C

hex_agent.py: starts training of agent --> update version number

hex_engine_example_script.py: starts agent vs human (not working atm)

test_machine.py: Implements the choosing of an action from a2c agent (not working atm)

v1_hex_actor.a2c: Agent trained against random player

v1_hex_episode_rewards.a2c: list of episode rewards from training of v1_hex_actor.a2c

v1_reward_hex.png: reward plot from training of v1_hex_actor.a2c

same for v2_...: Trained on random (10%) and vs v1
Agent is always trained 10% against random and 90% against a random pre version