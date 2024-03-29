# reil_hex

## Implementation of HEX agent with A2C

### Training of an agent
hex_agent.py: 
 - starts training of agent
 - training is by 10% against random and 90% against a random pre version as defined by global variables
 - set global variable:
   - AGENT_VERSION_NUMBER: the first version of the agent that should be trained
   - NUM_AGENTS_TO_TRAING: define how many agents should be trained
   - TRAIN_ON_LAST_NUM: define how many last versions should be trained against
   - MAX_EPISODES: define the maximum of episodes to train
   - EVAL_AFTER_X_EPISODES: define after how many episodes it will be evaluated if to stop training or further train
   - HEX_BOARD_SIZE: define the board size, mention, that previous versions need to have the same board size

### Files explanation
 - example_script.py: starts a game of the agent vs random, random vs agent and agent vs agent
 - v{x}_hex_actor.a2c: Trained agent, saved by joblib
 - v{x}_hex_episode_rewards.a2c: list of episode rewards from training of agent {x}, saved by joblib
 - v{x}_reward_hex.png: reward plot from training of agent {x}
 - actor_final_state_dict.torch:
   - state dict of the torch nn
   - used in test_machine.py
 - generate_torch_dict_from_joblib.py
   - converts the joblib agent to a torch state_dict
   - needed, as in test_machine.py joblib was somehow not working
 - fhtw_hex/submission/facade.py: 
   - implements the choosing of an action from a2c agent
   - uses last version of agent, must be transformed to state_dict beforehand with generate_torch_dict_from_joblib.py and copied to the folder fhtw_hex/submission
 - fhtw_hex/submission/actor_final_state_dict.torch:
   - state dict of the final actor
