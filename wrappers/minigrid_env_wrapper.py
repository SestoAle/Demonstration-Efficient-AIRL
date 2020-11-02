from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

import gym_minigrid
import argparse
from tensorforce.environments import Environment
import matplotlib.pyplot
import time

eps = 1e-12

class MinigridEnvWrapper(Environment):

    def __init__(self, game_name = 'MiniGrid-MultiRoom-N2-S4-v0', no_graphics = True,
                 with_channels = False,
                 with_local=True, sample_env = None,
                 num_envs = None,
                 rotate_env = False,
                 manual_input=False,verbose=False, terminate_env = False,
                 one_hot = False, input_channels = 3,
                 reward_model=None, _max_episode_timesteps=30,
                 ):

        self.with_local = with_local
        self.with_channels = with_channels
        self.one_hot = one_hot
        self.input_channels = input_channels
        self.manual_input = manual_input
        self.verbose = verbose
        self.reward_model = reward_model
        self._max_episode_timesteps = _max_episode_timesteps
        self.sample_env = sample_env
        self.num_envs = num_envs
        self.rotate_env = rotate_env
        self.terminate_env = terminate_env
        self.no_graphics = no_graphics
        self.probabilities = []
        self.no_graphics = no_graphics
        self.game_name = game_name
        if self.sample_env <= 0:
            self.sample_env = None



        self.minigrid = self.open_environment()

        self.current_id = 0

    def get_input_observation(self, state, action=None):

        observation = dict()

        if self.with_channels:
            if self.one_hot:
                global_in_one_hot = self.to_one_hot(state['image'][:, :, 0], 11)
                for i in range(1, self.input_channels):
                    global_in_one_hot = np.append(global_in_one_hot, self.to_one_hot(state['image'][:, :, i], 11), axis=2)
                global_in = global_in_one_hot
            else:
                global_in = state['image']
        else:
            global_in = state['image'][:,:,0]

        observation['global_in'] = global_in

        if self.with_local:
            if self.with_channels:
                local_in = state['local']
            else:
                local_in = state['local'][:,:,0]

            observation['local_in'] = local_in

        return observation

    def print_observation(self, observation, actions = None, reward = None):
        try:
            print(observation['global_in'])
            print('action = ' + str(actions))
            print('reward = ' + str(reward))
        except Exception as e:
            pass

    def command_to_action(self, command):

        switcher = {
            "q": 0,
            "e": 1,
            "w": 2,
            " ": 3,
        }

        return switcher.get(command, 99)

    def action_remapper(self, command):

        switcher = {
            0: 0,
            1: 1,
            2: 2,
            3: 5
        }

        return switcher.get(command, 99)


    def execute(self, actions):

        if self.manual_input:
            input_action = input('action: ')

            try:
                actions = input_action
            except ValueError:
                pass

        if isinstance(actions, str):
            actions = self.command_to_action(actions)
            #print(actions)

        actions = self.action_remapper(actions)

        observation, reward, done, info = self.minigrid.step(actions)
        if not self.no_graphics:
            self.minigrid.render('human')

        done = False
        if self.terminate_env:
            done = True

        observation = self.get_input_observation(observation)

        if self.verbose:
            self.print_observation(observation, actions, reward)

        return [observation, done, reward]

    def seed(self, seed):
        self.minigrid.seed(int(seed))

    def reset(self):

        if self.num_envs is not None:
            if self.rotate_env:
                seed = self.num_envs[(self.current_id + 1) % len(self.num_envs)]
                self.current_id += 1
            else:
                seed = self.num_envs[np.random.randint(0,len(self.num_envs))]
      
            self.minigrid.seed(seed)

        elif self.sample_env is not None:
            if self.rotate_env:
                seed = (self.current_id + 1) % self.sample_env
                self.current_id += 1
            else:
                seed = np.random.randint(0, self.sample_env)

            self.minigrid.seed(seed)

        observation = self.minigrid.reset()
        if not self.no_graphics:
            self.minigrid.render('human')

        observation = self.get_input_observation(observation)

        if self.verbose:
            self.print_observation(observation)

        return observation

    def open_environment(self):
        return FullyObsWrapper(gym.make(self.game_name))

    def re_open_environment(self):
        self.close()
        self.minigrid = self.open_environment()

    def close(self):
        matplotlib.pyplot.close('all')
        self.minigrid.close()

    def states(self):

        states = dict()

        if self.with_channels:
            if self.one_hot:
                states['global_in'] = {'shape': (15, 15, 33), 'type': 'float'}
            else:
                states['global_in'] = {'shape': (25, 25, 3), 'type': 'float'}
        else:
            states['global_in'] ={'shape': (15, 15), 'num_values': 11, 'type': 'int'}

        if self.with_local:
            if self.with_channels:
                states['local_in'] = {'shape': (7, 7, 3), 'type': 'float'}
            else:
                states['local_in'] = {'shape': (7, 7), 'num_values': 11, 'type': 'int'}

        return states

    def actions(self):
        return {
                    'type': 'int',
                    'num_values': 4
                }

    def add_probs(self, probs):
        self.probabilities.append(probs[0])

    def get_last_entropy(self):
        entropy = 0
        for prob in self.probabilities[-1]:
            entropy += (prob + eps)*(math.log(prob + eps) + eps)

        return -entropy

    def max_episode_timesteps(self):
        return self._max_episode_timesteps

    def set_config(self, config):
        return

    def to_one_hot(self, a, channels):
        return (np.arange(channels) == a[..., None]).astype(float)
