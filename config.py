# Environment to play
# environment = 'LunarLanderContinuous-v2'
environment = 'Pendulum-v0'
# environment = 'CartPole-v1'
# environment = 'Acrobot-v1'
# environment = 'MountainCar-v0'

# Continuous Action if true
# constinuous_action = False
continuous_action = True

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
