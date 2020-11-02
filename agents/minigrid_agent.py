from tensorforce.agents import Agent

def create_minigrid_agent(net, baseline, states, actions, args):


    agent = Agent.create(
        # Agent type
        agent='ppo',
        # Inputs structure
        states=states,
        # Actions structure
        actions=actions,
        network=net,
        # MemoryModel

        # 10 episodes per update
        batch_size=int(args.update_number),
        # Every 10 episodes
        update_frequency=int(args.update_number),
        max_episode_timesteps=int(args.num_timesteps),

        # DistributionModel

        discount=0.9,
        entropy_regularization=0.0,
        likelihood_ratio_clipping=0.5,

        critic_network=baseline,

        critic_optimizer=dict(
            type='multi_step',
            optimizer=dict(
                type='subsampling_step',
                fraction=0.33,
                optimizer=dict(
                    type='adam',
                    learning_rate=5e-4,
                )
            ),
            num_steps=10
        ),

        # PPOAgent

        learning_rate=5e-5,

        subsampling_fraction=0.33,
        optimization_steps=20,
        # TODO: check this part
        execution=None,
        # TensorFlow etc
        name='agent', device=None, parallel_interactions=1, seed=None, saver=None,
        summarizer=None, recorder=None
    )

    return agent
