import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['KMP_DUPLICATE_LIB_OK']="True"
from sys import platform

from agents.minigrid_agent import create_minigrid_agent
from agents.deepcrawl_agent import create_deepcrawl_agent
from reward_model.utils import NumpyEncoder
from reward_model.reward_model import AIRL
from reward_model.architectures.architectures import *

import time

import json

from wrappers.unity_env_wrapper import UnityEnvWrapper
from wrappers.minigrid_env_wrapper import MinigridEnvWrapper
from runners.custom_runner import CustomRunner

import argparse

use_cuda = True
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

'''---------'''
'''Arguments'''
'''---------'''

parser = argparse.ArgumentParser()

parser.add_argument('-mn', '--model-name', help="The name of the model", default="")
parser.add_argument('-ne', '--num-episodes', help="Specify the number of episodes after which the environment is restarted", default=None)
parser.add_argument('-wk', '--worker-id', help="The id for the worker", default=0)

parser.add_argument('-un', '--update-number', help="Update dict", default=10)
parser.add_argument('-ws', '--with-stats', help="Update dict", default=True)
parser.add_argument('-sa', '--save', help="Number of episodes for saving models", default=500)

parser.add_argument('-tn', '--task-name', help="The name of the environment", choices=['potions', 'minigrid'], default='potions')

# IRL
parser.add_argument('-nd', '--num-discriminator-exp', help="the number of episodes before update the reward model", default=10)
parser.add_argument('-rm','--reward-model', help="The name of the fixed reward model to use", default=None)
parser.add_argument('-se','--sampled-env', help="The number of Seed Levels to use", default=0)
parser.add_argument('-fr','--fixed-reward-model', dest='fixed_reward_model', help='Whether to use a fixed reward model' ,action='store_true')
parser.set_defaults(get_demonstrations=False)
parser.set_defaults(fixed_reward_model=False)

args = parser.parse_args()

'''--------------------'''
'''Algorithm Parameters'''
'''--------------------'''

task_name = args.task_name
# Choose the task to use
# Potions task
if task_name == 'potions':
    # Hyper-parameters for Potions task
    args.num_timesteps = 20
    irl_buffer_length = 100000
    del_mode = 'random_del'
    args.num_policy_update = 30
    args.demonstrations_name = 'dems_potions.pkl'

    if int(args.sampled_env) > 0:
        game_name = 'envs/DeepCrawl-Potions-SeedEnv'
    else:
        game_name = 'envs/DeepCrawl-Potions-ProcEnv'

    # Import net structures
    from net_structures.net_structures import dc2_net_conv_with_different_stats_cacca as net
    from net_structures.net_structures import dc2_net_conv_with_different_stats_cacca as baseline
    # Work ID of the environment. To use the unity editor, the ID must be 0. To use more environments in parallel, use
    # different ids
    work_id = int(args.worker_id)
    # Open the environment with all the desired flags
    environment = UnityEnvWrapper(game_name, no_graphics=False, seed=int(time.time()),
                                  worker_id=work_id, with_stats=args.with_stats, size_stats=18,
                                  size_global=10, agent_separate=False, with_class=False, with_hp=False,
                                  verbose=False, manual_input=False, _max_episode_timesteps=args.num_timesteps
                                  )
    # Create a Proximal Policy Optimization agent
    agent = create_deepcrawl_agent(net, baseline, environment.states(), environment.actions(), args)

    # Environment hyper-parameters
    curriculum = {
        'current_step': 0,
        'thresholds': [100e12],
        'parameters':
            {
                'minTargetHp': [1],
                'maxTargetHp': [1],
                'minAgentHp': [20],
                'maxAgentHp': [20],
                'minNumLoot': [0.2],
                'maxNumLoot': [0.2],
                'numActions': [17],
                # Agent statistics
                'agentAtk': [3],
                'agentDef': [3],
                'agentDes': [3],
                'sampledEnv': [int(args.sampled_env)],
                'rotateEnv': [False],
                'noAnim': [True]
            }
    }
# Minigrid task
elif task_name == 'minigrid':
    # Hyper-parameters for Minigrid task
    args.num_timesteps = 30
    irl_buffer_length = 630000
    del_mode = 'prob_del'
    game_name = "MiniGrid-MultiRoom-N3-S5-v0"
    args.demonstrations_name = 'dems_minigrid.pkl'

    args.num_policy_update = 30
    # Import net structures
    from net_structures.minigrid_structures import minigrid as net
    from net_structures.minigrid_structures import minigrid as baseline
    environment = MinigridEnvWrapper(game_name="MiniGrid-MultiRoom-N3-S5-v0", manual_input=False, verbose=False,
                                     no_graphics=False, sample_env=int(args.sampled_env))

    # Create a Proximal Policy Optimization agent
    agent = create_minigrid_agent(net, baseline, environment.states(), environment.actions(), args)

    curriculum = None



# Number of episodes of a single run
num_episodes = args.num_episodes
# Number of timesteps within an episode
num_timesteps = args.num_timesteps

model_name = args.model_name

'''-----------------'''
'''Run the algorithm'''
'''-----------------'''

use_model = None

if args.model_name is not '':
    model_name = args.model_name
else:
    model_name = input('Specify the name to save policy and reward model: ')
history = {
    "episode_rewards": list(),
    "episode_timesteps": list(),
    "mean_entropies": list(),
    "std_entropies": list(),
    "reward_model_loss": list(),
    "env_rewards": list()
}


start_time = time.time()

# Callback function printing episode statistics
def episode_finished(r, c, episode = 100):
    if(len(r.history['episode_rewards']) % episode == 0):
        print('Average cumulative reward for ' + str(episode) + ' episodes @ episode ' + str(len(r.history['episode_rewards'])) + ': ' + str(np.mean(r.history['episode_rewards'][-episode:])))
        print('The agent made ' + str(sum(r.history['episode_timesteps'])) + ' steps so far')

        # IRL stat
        print('Average cumulative env reward for ' + str(episode) + ' episodes @ episode ' + str(len(r.history['env_rewards'])) + ': ' + str(np.mean(r.history['env_rewards'][-episode:])))
        if r.reward_model is not None and not args.fixed_reward_model:
            print('Reward Model Loss @ episode {}: {}'.format(episode, np.mean(r.history['reward_model_loss'][-episode:])))

        timer(start_time, time.time())

    # If num_episodes is not defined, save the model every 3000 episodes
    if(num_episodes == None):
        save = int(args.save)
    else:
        save = num_episodes

    if(len(r.history['episode_rewards']) % save == 0):
      save_model(r)

    return True

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def save_model(runner):
    history = runner.history
    # Save the runner statistics
    # Save the model and the runner statistics
    runner.agent.save(directory='saved', filename=model_name)

    # If IRL, save reward model
    if reward_model is not None and not args.fixed_reward_model:
        reward_model.save_model(model_name + "_" + str(len(history['episode_timesteps'])))

    json_str = json.dumps(history, cls=NumpyEncoder)
    f = open("arrays/" + model_name + ".json", "w")
    f.write(json_str)
    f.close()

    runner.reset()

try:

    # Create the reward model with all hyper-parameters
    reward_model = None

    directory = os.path.join(os.getcwd(), "saved/")
    while not os.path.isfile(directory + '{}.meta'.format(model_name)):
        print('Model not found')
        model_name = input('Specify the name to save policy and reward model: ')

    agent.restore(directory, model_name)


    # Create the runner to run the algorithm
    runner = CustomRunner(agent=agent, max_episode_timesteps=args.num_timesteps, environment=environment,
                          history=history,
                          # Reward Model
                          reward_model=reward_model, model_name=model_name,
                          num_discriminator_exp=int(args.num_discriminator_exp),
                          num_policy_updates=int(args.num_policy_update),
                          update_number=int(args.update_number),
                          fixed_reward_model=args.fixed_reward_model,
                          sampled_env = int(args.sampled_env),
                          dems_name=args.demonstrations_name,
                          irl_buffer_length=irl_buffer_length,
                          curriculum=curriculum,
                          irl_del_mode=del_mode
                          )

    runner.run(num_episodes=None, callback=episode_finished, use_tqdm=False)

finally:

    '''--------------'''
    '''End of the run'''
    '''--------------'''

    # Save the model and the runner statistics
    if model_name == "" or model_name == " " or model_name == None:
        saveName = input('Do you want to specify a save name? [y/n] ')
        if(saveName == 'y'):
            saveName = input('Specify the name ')
        else:
            saveName = 'Model'
    else:
        saveName = model_name

    save_model(runner)

    # Close the runner
    runner.close()

    print("Model saved with name " + saveName)
