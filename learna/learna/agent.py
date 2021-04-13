from tensorforce.agents import PPOAgent, RandomAgent
from dataclasses import dataclass

from functools import partial


@dataclass
class NetworkConfig:
    """
    Dataclass providing the network configuration.

    Default values describe:
        conv_sizes: The number of filters.
        conv_channels: The convolution window size.
        lstm_units: The number of lstm units in a single layer.
        fc_units: The number of units in a single fully-connected layer.
        fc_activation: The activation function.
        num_fc_layers: The number of fully-connected layers.
        embedding_size: The dimensionality of the embedding layer.
    """

    conv_sizes: int = (3, 2)
    conv_channels: int = (16, 8)
    lstm_units: int = 1
    num_lstm_layers: int = 0
    fc_units: int = 2
    fc_activation: str = "relu"
    num_fc_layers: int = 2
    embedding_size: int = 0


def get_network(network_config):
    """
    Get a specific policy network as specified in the network configuration.

    Args:
        network_config: The configuration for the nertwork.

    Returns:
        The policy network of the agent.
    """
    embedding = [
        dict(
            type="embedding",
            indices=4,
            size=network_config.embedding_size,
            l2_regularization=0.0,
            l1_regularization=0.0,
        )
    ]
    convolution = [
        dict(
            type="conv1d",
            size=size,
            window=window,
            stride=1,
            padding="VALID",
            bias=True,
            activation="relu",
            l2_regularization=0.0,
            l1_regularization=0.0,
        )
        for size, window in zip(network_config.conv_channels, network_config.conv_sizes)
        if window > 1
    ]
    lstm = [dict(type="internal_lstm", size=network_config.lstm_units)]
    dense = [
        dict(
            type="dense",
            size=network_config.fc_units,
            bias=True,
            activation=network_config.fc_activation,
            l2_regularization=0.0,
            l1_regularization=0.0,
        )
    ]

    use_conv = any(map(lambda x: x > 1, network_config.conv_sizes))
    network = []
    if network_config.embedding_size:
        network += embedding
    if use_conv:
        network += convolution
    if use_conv or network_config.embedding_size:
        network += [dict(type="flatten")]
    network += network_config.num_lstm_layers * lstm
    network += network_config.num_fc_layers * dense

    return network


@dataclass
class AgentConfig:
    """
    Dataclass providing the agent configuration.

    Default values describe:
        learning_rate: The learning rate to use by PPO.
        batch_size: Integer of the batch size.
        optimization_steps: The number of optimization steps.
        likelihood_ratio_clipping: Likelihood ratio clipping for policy gradient.
        entropy_regularization: Entropy regularization weight.
        random_agent: Defines if agent is a random agent.
    """

    learning_rate: float = 5e-4
    batch_size: int = 5
    likelihood_ratio_clipping: float = 0.3
    entropy_regularization: float = 1.5e-3
    random_agent: bool = False


# This is needed / used because of the threaded runner interface of tensorforce, see
# learn_to_design_rna.py.
def ppo_agent_kwargs(agent_config, session_config):
    """
    Get keyword arguments for initializing a PPO agent.

    Args:
        agent_config: The configuration of the agent.
        session_config: The session configuration.

    Returns:
        Dictionary of arguments for initialization of a PPO agent.
    """
    step_optimizer = dict(type="adam", learning_rate=agent_config.learning_rate)
    return dict(
        device=None,
        session_config=session_config,
        distributed_spec=None,
        distributions_spec=None,
        discount=1.0,
        batched_observe=1000,  # set to batch_size?
        batch_size=agent_config.batch_size,
        keep_last_timestep=True,
        step_optimizer=step_optimizer,
        optimization_steps=1,
        likelihood_ratio_clipping=agent_config.likelihood_ratio_clipping,
        entropy_regularization=agent_config.entropy_regularization,
        variable_noise=None,
        states_preprocessing_spec=None,
        explorations_spec=None,
        reward_preprocessing_spec=None,
        baseline_mode=None,
        baseline=None,
        baseline_optimizer=None,
        gae_lambda=None,
    )


# This is needed / used because of the threaded runner interface of tensorforce, see
# learn_to_design_rna.py.
def random_agent_kwargs(agent_config, session_config):
    """
    Get a random agent's keyword arguments for initialization.

    Args:
        agent_config: The configuration of the agent.
        session_config: The session configuration.

    Returns:
        A dictionary of keyword args for a random agent.
    """
    return dict(
        device=None,
        session_config=session_config,
        distributed_spec=None,
        discount=1.0,
        batched_observe=1000,
        variable_noise=None,
        states_preprocessing_spec=None,
        explorations_spec=None,
        reward_preprocessing_spec=None,
    )


def get_agent(environment, network, agent_config, session_config, restore_path):
    """
    Get an agent. General function providing agents.

    Args:
        environment: The environment.
        network: The network of the agent.
        agent_config: The configuration of the agent.
        session_config: The configuration of the session.
        restore_path: Path to restore saved data from.

    Returns:
       An agent.
    """
    if agent_config.random_agent:
        return RandomAgent(
            environment.states,
            environment.actions,
            **random_agent_kwargs(agent_config, session_config)
        )

    agent = PPOAgent(
        environment.states,
        environment.actions,
        network,
        **ppo_agent_kwargs(agent_config, session_config)
    )
    if restore_path:
        agent.restore_model(directory=restore_path)
    return agent


def get_agent_fn(**kwargs):
    """
    Function to generically get agents using keyword arguments <**kwargs>.
    """
    return partial(get_agent, **kwargs)
