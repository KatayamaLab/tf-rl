# Environment to play

from gym.envs.registration import register

register(
    id='SwingUpCartPole-v0',
    entry_point='swingupcartpole:SwingUpCartPoleEnv',
    max_episode_steps = 500,
    reward_threshold=475.0,
)

import gym


# Continuous action environment
# environment = 'LunarLanderContinuous-v2'
# environment = 'Pendulum-v0'

# Descrete action environment
# environment = 'CartPole-v1'
environment = 'SwingUpCartPole-v0'

# environment = 'Acrobot-v1'
# environment = 'MountainCar-v0'

# Continuous Action if true
# continuous_action = False
continuous_action = False

# Number of Episodes
max_episodes = 100000

#Summaries log directory (/tmp/tf-rl/log)
log_dir = '/tmp/tf-rl/log'

# Model path for save (/tmp/tf-rl/model/)')
save_model_dir = '/tmp/tf-rl/model/'

#Interval to save model (500)
interval_to_save_model =  500

# Model path for restore
restore_model_path =  None
#restore_model_path =  '/tmp/tf-rl/model/model-500'

# Training mode
train = True

# Render mode
render = True

# Record mode
record = False

# Record path
record_path = 'record/'
