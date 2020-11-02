from tensorforce.execution.runner import Runner
import numpy as np
import time
import random

class CustomRunner(Runner):

    def __init__(self, agent, environment, max_episode_timesteps=100, history=None, curriculum = None,
                 reward_model = None, num_policy_updates = 5, num_discriminator_exp = 5, save_experience = False,
                 model_name = None, dems_name = 'dems.pkl', reward_array = None, reward_alpha_array = None,
                 testing = False, fixed_reward_model = False, update_number = 10, sampled_env=20,
                 # IRL buffer
                 irl_buffer_length = 630000, irl_del_mode='random_del',
                 # Adversarial
                 use_double_agent = False, double_agent = None,
                 evaluation=False
                 ):
        self.mean_entropies = []
        self.std_entropies = []
        self.history = history
        self.curriculum = curriculum
        self.start_training = 0
        self.unity_env = environment
        self.max_episode_timesteps = max_episode_timesteps

        self.env_rewards = []
        self.reward_model_loss = []
        self.reward_model_val_loss = []
        self.reward_model = reward_model
        self.num_policy_updates = num_policy_updates
        self.num_discriminator_exp = num_discriminator_exp
        self.update_number = update_number
        self.save_experience = save_experience
        self.current_curriculum_step = 0
        self.dems_name = dems_name
        self.reward_array = reward_array
        self.reward_alpha_array = reward_alpha_array

        # Self learning curriculum
        self.model_name = model_name
        self.use_double_agent = use_double_agent
        self.double_agent = double_agent

        # Wether this is a testing phase
        self.testing = testing
        # Wheter runner uses a fixed reward model or it must be learnt
        self.fixed_reward_model = fixed_reward_model

        self.irl_buffer_length = irl_buffer_length
        self.irl_del_mode = irl_del_mode

        # DeepCrawl
        if not isinstance(environment,list):
            environment = self.unity_env
            environments = None
            self.unity_env = [self.unity_env]
        else:
            environment = None
            environments = self.unity_env

        self.local_entropies = np.empty((len(self.unity_env), 0)).tolist()
        self.env_episode_reward = np.zeros((len(self.unity_env)))

        # IRL
        self._reset = True
        self.initial_states = []
        self._rewards = {}
        self._discs = {}


        super(CustomRunner, self).__init__(agent, environment=environment, environments=environments, max_episode_timesteps=max_episode_timesteps, evaluation=evaluation)

        if testing:
            self.evaluation_internals = self.agent.initial_internals()

        # DeepCrawl
        if self.curriculum is not None:
            for env in self.unity_env:
                config = self.set_curriculum(self.curriculum, np.sum(self.history['episode_timesteps']))
                if self.start_training == 0:
                    print(config)
                env.set_config(config)
        # DeepCrawl

        # Initialize Reward Model, if IRL
        if self.reward_model is not None and not testing and not fixed_reward_model:
            if len(self.unity_env) > 1:
                print("For Inverse Reinforcement Learning you can't use parallel environments")
                return

            y = None

            while y != 'y' and y != 'n' and y != 'p':
                y = input('Do you want to create new demonstrations? [Y(es)/N(o))]\n'
                          '[N] will load demonstrations made by an expert policy\n')
                y = y.lower()
                y = y.replace('es','')
                y = y.replace('o', '')
                if y == 'y' or y == 'p':

                    if curriculum is not None:
                        self.curriculum['parameters']['rotateEnv'] = [True]
                        self.curriculum['parameters']['noAnim'] = [False]
                        config = self.set_curriculum(self.curriculum, 0)
                        self.unity_env[0].set_config(config)
                    else:
                        self.unity_env[0].rotate_env = True

                    self.unity_env[0].no_graphics = False
                    self.unity_env[0].re_open_environment()

                if y == 'y':
                    dems, vals = self.reward_model.create_demonstrations(env=self.unity_env[0], max_timestep=max_episode_timesteps, dems_name=self.dems_name, sampled_env=sampled_env)
                #elif y == 'p':
                    #dems, vals = self.reward_model.create_demonstrations(env=self.unity_env[0], max_timestep=max_episode_timesteps, with_policy=True, dems_name=self.dems_name)
                elif y=='n':
                    print('Loading demonstrations...')
                    dems, vals = self.reward_model.load_demonstrations()


            if y == 'y' or y=='p':
                self.unity_env[0].no_graphics = True
                self.unity_env[0].re_open_environment()
                if curriculum is not None:
                    config = self.set_curriculum(self.curriculum, np.sum(self.history['episode_timesteps']))
                    self.unity_env[0].set_config(config)
                    self.curriculum['parameters']['rotateEnv'] = [False]
                    self.curriculum['parameters']['noAnim'] = [True]
                else:
                    self.unity_env[0].rotate_env = False

            print('Demonstrations loaded! We have ' + str(len(dems['obs'])) + " timesteps in these demonstrations")

            # IRL data structures
            self.policy_buffer = {
                'obs': [],
                'obs_n': [],
                'acts': [],
            }

            self.policy_traj = {
                'obs': [],
                'obs_n': [],
                'acts': [],
            }

            self.irl_states = []
            self.irl_acts = []
            self.irl_probs = []

            self.policy_episodes = []

            print('Getting initial experience...')
            self.policy_traj = self.get_experience(max_episode_timesteps, self.unity_env[0], int(self.num_policy_updates / self.update_number * self.num_discriminator_exp), random=True)

            self.update_reward_model()

        # Start the training
        self.start_training = 1
        #self.reset(history)

    # DeepCrawl: Update curriculum
    def set_curriculum(self, curriculum, total_timesteps, mode='steps'):

        if curriculum == None:
            return None

        if mode == 'steps':
            lessons = np.cumsum(curriculum['thresholds'])

            curriculum_step = 0

            for (index, l) in enumerate(lessons):
                if total_timesteps > l:
                    curriculum_step = index + 1

        # TODO: DA FARE ASSOLUTAMENTE CURRICULUM CON MEDIA
        elif mode == 'mean':
            if len(self.episode_rewards) <= 100 * 6:
                self.current_curriculum_step = 0
                pass

            means = []
            for i in range(6):
                mean = np.mean(self.episode_rewards[:-100 * (i + 1)])
                means.append(mean)

            mean = np.mean(np.asarray(means))
            if mean - curriculum['lessons'][self.current_curriculum_step] < 0.05:
                self.current_curriculum_step += 1

            config = {}
            parameters = curriculum['parameters']
            for (par, value) in parameters.items():
                config[par] = value[self.current_curriculum_step]

        parameters = curriculum['parameters']
        config = {}

        for (par, value) in parameters.items():
            config[par] = value[self.current_curriculum_step]

        # Self curriculum setting:
        # Save the model
        # TODO: FARE SELF CURRICULUM PLAY
        if self.use_double_agent:
            if curriculum_step > self.current_curriculum_step:
                self.agent.save(directory='saved/adversarial', filename=self.model_name + '_' + str(curriculum_step))
                self.double_agent.restore(directory='saved/adversarial', filename=self.model_name + '_' + str(curriculum_step))

        self.current_curriculum_step = curriculum_step

        return config

    def handle_act(self, parallel):

        # If first act after the reset, save the reset state
        if self.reward_model is not None and not self.fixed_reward_model:
            if len(self.irl_states) <= 0:
                self.irl_states.append(self.states[parallel])

        # If first act after the reset
        if self._reset:
            self.initial_states.append(self.states[parallel])
            self._reset = False

        if self.batch_agent_calls:
            self.environments[parallel].start_execute(actions=self.actions[parallel])

        else:
            agent_start = time.time()

            # DeepCrawl
            query = ['action-distribution-probabilities']
            if not self.testing:
                actions, probs = self.agent.act(states=self.states[parallel], parallel=parallel, query=query)
            else:
                actions, self.evaluation_internals, probs = self.agent.act(
                    states=self.states[-1], internals=self.evaluation_internals, evaluation=True, query = query
                )
            self.irl_state = self.states[parallel]
            probs = probs[0]
            self.irl_c_probs = probs
            self.actions = actions

            self.unity_env[parallel].add_probs(probs)
            self.local_entropies[parallel].append(self.unity_env[parallel].get_last_entropy())
            # DeepCrawl
            self.episode_agent_second[parallel] += time.time() - agent_start

            self.environments[parallel].start_execute(actions=actions)

        # Update episode statistics
        self.episode_timestep[parallel] += 1

        # Maximum number of timesteps or timestep callback (after counter increment!)
        self.timesteps += 1
        if (
            (self.episode_timestep[parallel] % self.callback_timestep_frequency == 0 and not self.callback(self)) or
            self.timesteps >= self.num_timesteps
        ):
            self.terminate = 2

    def handle_observe(self, parallel):

        # IRL settings
        if self.reward_model is not None or self.reward_array is not None:

            # reward_from_model = self.reward_model.forward([state], [action])
            # reward_from_model = self.reward_model.forward([state], [action])
            probs = np.squeeze(self.irl_c_probs)
            state = self.irl_state
            actions = self.actions
            state_n = self.states[parallel]
            reward_from_model = 0
            if not self.fixed_reward_model:
                reward_from_model, _, _, _ = self.reward_model.eval_discriminator([state], [state_n], [probs[actions]],
                                                                         [actions])
            else:
                if self.reward_array is not None:
                    for (index, model) in enumerate(self.reward_array):
                        model_rew = model.eval([state], [actions])
                        model.push_reward(model_rew)
                        model_rew = model.normalize_rewards(model_rew)
                        model_rew *= self.reward_alpha_array[index]
                        reward_from_model += model_rew

                else:
                    reward_from_model = self.reward_model.eval([state], [state_n], [actions], probs=[probs[actions]])

            reward_from_model = np.squeeze(reward_from_model)

            if not self.fixed_reward_model:
                self.irl_states.append(state_n)
                self.irl_acts.append(self.actions)
                self.irl_probs.append(probs[self.actions])

            self.env_episode_reward[parallel] += self.rewards[parallel]
            self.rewards[parallel] = reward_from_model
            #self.rewards[parallel] += reward_from_model

        # Update episode statistics
        self.episode_reward[parallel] += self.rewards[parallel]

        # Not terminal but finished
        if self.terminals[parallel] == 0 and self.terminate == 2:
            self.terminals[parallel] = 2

        # Observe unless batch_agent_calls
        if not self.testing and not self.batch_agent_calls:
            agent_start = time.time()
            updated = self.agent.observe(
                terminal=self.terminals[parallel], reward=self.rewards[parallel], parallel=parallel
            )
            self.episode_agent_second[parallel] += time.time() - agent_start
            self.updates += int(updated)

        # Maximum number of updates (after counter increment!)
        if self.updates >= self.num_updates:
            self.terminate = 2

    def handle_terminal(self, parallel):
        # Update experiment statistics
        self.episode_rewards.append(self.episode_reward[parallel])
        self.episode_timesteps.append(self.episode_timestep[parallel])
        self.episode_seconds.append(time.time() - self.episode_start[parallel])
        self.episode_agent_seconds.append(self.episode_agent_second[parallel])
        # DeepCrawl
        self.mean_entropies.append(np.mean(self.local_entropies[parallel]))
        self.std_entropies.append(np.std(self.local_entropies[parallel]))
        # IRL stat
        self.env_rewards.append(self.env_episode_reward[parallel])
        # DeepCrawl

        # At the end of each episode, update IRL buffers
        if self.reward_model is not None and not self.testing and not self.fixed_reward_model:  # and\
            # self.global_episode % self.agent.update_mode['frequency'] == 0:
            # self.get_experience_(policy_traj, max_episode_timesteps, self.num_discriminator_exp)
            self.policy_traj['obs'] = self.irl_states[:-1]
            self.policy_traj['obs_n'] = self.irl_states[1:]
            self.policy_traj['acts'] = self.irl_acts
            self.policy_episodes.append(self.policy_traj)
            self.policy_traj = {
                'obs': [],
                'obs_n': [],
                'acts': [],
            }

        self.irl_states = []
        self.irl_acts = []
        self.irl_probs = []

        self.update_history()
        self.update_reward_model()

        # Maximum number of episodes or episode callback (after counter increment!)
        self.episodes += 1
        if self.terminate == 0 and ((
            self.episodes % self.callback_episode_frequency == 0 and
            not self.callback(self, parallel)
        ) or self.episodes >= self.num_episodes):
            self.terminate = 1

        # Reset episode statistics
        self.episode_reward[parallel] = 0.0
        # IRL stat
        self.env_episode_reward[parallel] = 0.0
        self.episode_timestep[parallel] = 0
        self.episode_agent_second[parallel] = 0.0
        self.episode_start[parallel] = time.time()

        # Reset environment
        if self.terminate == 0 and not self.sync_episodes:
            self.terminals[parallel] = -1
            # DeepCrawl
            # Set curriculum configuration
            for env in self.unity_env:
                config = self.set_curriculum(self.curriculum, np.sum(self.history['episode_timesteps']))
                if self.start_training == 0:
                    print(config)
                self.start_training = 1
                env.set_config(config)
            # DeepCrawl
            self.environments[parallel].start_reset()
            self._reset = True
            if self.testing:
                self.evaluation_internals = self.agent.initial_internals()

    def handle_terminal_evaluation(self):
        # Update experiment statistics
        self.evaluation_rewards.append(self.episode_reward[-1])
        self.evaluation_timesteps.append(self.episode_timestep[-1])
        self.evaluation_seconds.append(time.time() - self.evaluation_start)
        self.evaluation_agent_seconds.append(self.evaluation_agent_second)
        if self.is_environment_remote:
            self.evaluation_env_seconds.append(self.environments[-1].episode_seconds)

        # Evaluation callback
        if self.save_best_agent is not None:
            evaluation_score = self.evaluation_callback(self)
            assert isinstance(evaluation_score, float)
            if self.best_evaluation_score is None:
                self.best_evaluation_score = evaluation_score
            elif evaluation_score > self.best_evaluation_score:
                self.best_evaluation_score = evaluation_score
                self.agent.save(
                    directory=self.save_best_agent, filename='best-model', append=None
                )
        else:
            self.evaluation_callback(self)

        # Maximum number of episodes or episode callback (after counter increment!)
        if self.evaluation_run and len(self.environments) == 1:
            self.episodes += 1
            if self.terminate == 0 and ((
                self.episodes % self.callback_episode_frequency == 0 and
                not self.callback(self, 0)
            ) or self.episodes >= self.num_episodes):
                self.terminate = 1

        # Reset episode statistics
        self.episode_reward[-1] = 0.0
        self.episode_timestep[-1] = 0
        self.evaluation_agent_second = 0.0
        self.evaluation_start = time.time()

        # Reset environment
        if self.terminate == 0 and not self.sync_episodes:
            self.terminals[-1] = 0
            self.environments[-1].start_reset()
            self.evaluation_internals = self.agent.initial_internals()

    def update_history(self):
        self.history["episode_rewards"].extend(self.episode_rewards)
        self.history["episode_timesteps"].extend(self.episode_timesteps)
        self.history["mean_entropies"].extend(self.mean_entropies)
        self.history["std_entropies"].extend(self.std_entropies)

        self.history["reward_model_loss"].extend(self.reward_model_loss)
        self.history["env_rewards"].extend(self.env_rewards)
        self.reset()

    def reset(self):
        self.episode_rewards = list()
        self.episode_timesteps = list()
        self.std_entropies = list()
        self.mean_entropies = list()
        self.reward_model_loss = list()
        self.env_rewards = list()

    # IRL Setting
    # Get experience from environment, the agent act in the env without update itself
    def get_experience(self, max_episode_timesteps, env, num_discriminator_exp=None, verbose=False, random=False):

        if num_discriminator_exp == None:
            num_discriminator_exp = self.num_policy_updates

        policy_traj = {
            'obs': [],
            'obs_n': [],
            'acts': []
        }

        # For policy update number
        for ep in range(num_discriminator_exp):
            states = []
            probs = []
            actions = []
            state = env.reset()
            states.append(state)

            step = 0
            # While the episode si not finished
            reward = 0
            while True:
                step += 1
                # Get the experiences that are not saved in the agent
                query = ['action-distribution-probabilities']

                action, fetch = self.agent.act(states=state, deterministic=False, independent=True, evaluation=False,
                                               query=query)

                if random:
                    num_actions = env.actions()['num_values']
                    action = np.random.randint(0, num_actions)
                c_probs = np.squeeze(fetch[0])
                state_n, terminal, step_reward = env.execute(actions=action)

                # reward_from_model = self.reward_model.eval_discriminator([state], [state_n], [c_probs[action]],
                #                                                          [action])
                # self.reward_model.push_reward(reward_from_model)
                state = state_n

                reward += step_reward

                states.append(state)
                actions.append(action)
                probs.append(c_probs[action])

                if terminal or step >= max_episode_timesteps:
                    break

            if verbose:
                print("Reward at the end of episode " + str(ep + 1) + ": " + str(reward))

            # Saved the last episode experiences
            policy_traj['obs'].extend(states[:-1])
            policy_traj['obs_n'].extend(states[1:])
            policy_traj['acts'].extend(actions)

        # Return all the experience
        return policy_traj

    # Made some train steps of the reward model
    def update_reward_model(self):
        if self.reward_model is not None and not self.testing and not self.fixed_reward_model:
            if len(self.history['episode_rewards']) % self.num_policy_updates == 0:

                if len(self.history['episode_rewards']) != 0 and self.start_training == 1:
                    for i in range(self.num_discriminator_exp):
                        policy_traj = self.get_experience(self.max_episode_timesteps, self.unity_env[0], 1)
                        self.policy_episodes.append(policy_traj)

                    self.policy_traj = self.policy_episode_to_policy_traj(self.policy_episodes)

                self.add_to_poilicy_buffer(self.policy_buffer, self.policy_traj, buffer_length=self.irl_buffer_length, del_mode=self.irl_del_mode)

                # loss, val_loss = self.reward_model.train_step(self.reward_model.expert_traj, policy_traj)
                loss, val_loss = self.reward_model.fit(self.reward_model.expert_traj, self.policy_buffer)
                self.reward_model_loss.append(loss)
                self.reward_model_val_loss.append(val_loss)
                self.policy_traj = {
                    'obs': [],
                    'obs_n': [],
                    'acts': [],
                }
                self.policy_episodes = []
    
    # Update policy buffer of IRL
    def add_to_poilicy_buffer(self, policy_buffer, new_policy_buffer, buffer_length=630000, del_mode='fifo'):
 
        if len(policy_buffer['obs']) + len(new_policy_buffer['obs']) > buffer_length:
            diff = len(policy_buffer['obs']) + len(new_policy_buffer['obs']) - buffer_length

            if del_mode == 'prob_del':
                self.add_paths(policy_buffer, diff)
                policy_buffer['obs'] = policy_buffer['obs'].tolist()
                policy_buffer['obs_n'] = policy_buffer['obs_n'].tolist()
                policy_buffer['acts'] = policy_buffer['acts'].tolist()
            elif del_mode == 'random_del':
                indexes = random.sample(range(len(policy_buffer['obs'])), diff)
                policy_buffer['obs'] = np.delete(policy_buffer['obs'], indexes)
                policy_buffer['obs_n'] = np.delete(policy_buffer['obs_n'], indexes)
                policy_buffer['acts'] = np.delete(policy_buffer['acts'], indexes)
                policy_buffer['obs'] = policy_buffer['obs'].tolist()
                policy_buffer['obs_n'] = policy_buffer['obs_n'].tolist()
                policy_buffer['acts'] = policy_buffer['acts'].tolist()
            else:
                policy_buffer['obs'] = policy_buffer['obs'][diff:]
                policy_buffer['obs_n'] = policy_buffer['obs_n'][diff:]
                policy_buffer['acts'] = policy_buffer['acts'][diff:]


        policy_buffer['obs'].extend(new_policy_buffer['obs'])
        policy_buffer['obs_n'].extend(new_policy_buffer['obs_n'])
        policy_buffer['acts'].extend(new_policy_buffer['acts'])

    def add_paths(self, policy_buffer, overflow):

        while overflow > 0:
            # self.buffer = self.buffer[overflow:]
            N = len(policy_buffer['obs'])
            probs = np.arange(N) + 1
            probs = probs / float(np.sum(probs))
            pidx = np.random.choice(np.arange(N), p=probs)

            policy_buffer['obs'] = np.delete(policy_buffer['obs'], pidx)
            policy_buffer['obs_n'] = np.delete(policy_buffer['obs_n'], pidx)
            policy_buffer['acts'] = np.delete(policy_buffer['acts'], pidx)

            overflow -= 1

    def policy_episode_to_policy_traj(self, policy_episodes):
        policy_episodes = policy_episodes[self.num_discriminator_exp:]
        policy_traj = {
            'obs': [],
            'obs_n': [],
            'acts': []
        }
        for episode in policy_episodes:
            policy_traj['obs'].extend(episode['obs'])
            policy_traj['obs_n'].extend(episode['obs_n'])
            policy_traj['acts'].extend(episode['acts'])

        return policy_traj


