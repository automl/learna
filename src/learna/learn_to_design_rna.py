import multiprocessing
from pathlib import Path

from .agent import NetworkConfig, get_network, AgentConfig, ppo_agent_kwargs, get_agent
from .environment import RnaDesignEnvironment, RnaDesignEnvironmentConfig

from ..tensorforce.threaded_runner import clone_worker_agent, ThreadedRunner


def episode_finished(stats):
    """
    Function called after each episode of the agent (after designing an entire candidate solution).

    Args:
       stats: Statistics to be printed.

    Returns:
       True, meaning to continue running.
    """
    print(stats)
    return True


def learn_to_design_rna(
    dot_brackets,
    timeout,
    worker_count,
    save_path,
    restore_path,
    network_config,
    agent_config,
    env_config,
):
    """
    Main function for training the agent for RNA design. Instanciate agents and environments
    to run in a threaded runner using asynchronous parallel PPO.

    Args:
        TODO
        timeout: Maximum time to run.
        worker_count: The number of workers to run training on.
        save_path: Path for saving the trained model.
        restore_path: Path to restore saved configurations/models from.
        network_config: The configuration of the network.
        agent_config: The configuration of the agent.
        env_config: The configuration of the environment.

    Returns:
        Information on the episodes.
    """
    env_config.use_conv = any(map(lambda x: x > 1, network_config.conv_sizes))
    env_config.use_embedding = bool(network_config.embedding_size)
    environments = [
        RnaDesignEnvironment(dot_brackets, env_config) for _ in range(worker_count)
    ]

    network = get_network(network_config)
    agent = get_agent(
        environment=environments[0],
        network=network,
        agent_config=agent_config,
        session_config=None,
        restore_path=restore_path,
    )
    agents = clone_worker_agent(
        agent,
        worker_count,
        environments[0],
        network,
        ppo_agent_kwargs(agent_config, session_config=None),
    )
    threaded_runner = ThreadedRunner(agents, environments)
    # Bug in threaded runner requires a summary report
    threaded_runner.run(
        timeout=timeout, episode_finished=episode_finished, summary_report=lambda x: x
    )

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        agent.save_model(directory=save_path.joinpath("last_model"))

    episodes_infos = [environment.episodes_info for environment in environments]
    return episodes_infos


if __name__ == "__main__":
    import argparse
    from ..data.parse_dot_brackets import parse_dot_brackets

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--dataset", type=Path, help="Available: eterna, rfam_taneda")
    parser.add_argument(
        "--target_structure_ids",
        default=None,
        type=int,
        nargs="+",
        help="List of target structure ids to run on",
    )

    # Model
    parser.add_argument("--restore_path", type=Path, help="From where to load model")
    parser.add_argument("--save_path", type=Path, help="Where to save models")

    # Exectuion behaviour
    parser.add_argument("--timeout", type=int, help="Maximum time to run")
    parser.add_argument("--worker_count", type=int, help="Number of threads to use")

    # Hyperparameters
    parser.add_argument("--learning_rate", type=float, help="Learning rate to use")
    parser.add_argument(
        "--mutation_threshold", type=int, help="Enable MUTATION with set threshold"
    )
    parser.add_argument(
        "--reward_exponent", default=1, type=float, help="Exponent for reward shaping"
    )
    parser.add_argument(
        "--state_radius", default=0, type=int, help="Radius around current site"
    )
    parser.add_argument(
        "--conv_sizes", type=int, default=[1, 2], nargs="+", help="Size of conv kernels"
    )
    parser.add_argument(
        "--conv_channels",
        type=int,
        default=[50, 2],
        nargs="+",
        help="Channel size of conv",
    )
    parser.add_argument(
        "--num_fc_layers", type=int, default=2, help="Number of FC layers to use"
    )
    parser.add_argument(
        "--fc_units", type=int, default=50, help="Number of units to use per FC layer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for ppo agent"
    )
    parser.add_argument(
        "--entropy_regularization", type=float, default=1.5e-3, help="The output entropy"
    )
    parser.add_argument("--embedding_size", type=int, default=0, help="Size of embedding")
    parser.add_argument(
        "--lstm_units", type=int, default=0, help="Number of lstm units in each layer"
    )
    parser.add_argument(
        "--num_lstm_layers", type=int, default=0, help="Number of lstm layers"
    )

    args = parser.parse_args()

    network_config = NetworkConfig(
        conv_sizes=args.conv_sizes,
        conv_channels=args.conv_channels,
        num_fc_layers=args.num_fc_layers,
        fc_units=args.fc_units,
        embedding_size=args.embedding_size,
        lstm_units=args.lstm_units,
        num_lstm_layers=args.num_lstm_layers,
    )
    agent_config = AgentConfig(learning_rate=args.learning_rate)
    env_config = RnaDesignEnvironmentConfig(
        mutation_threshold=args.mutation_threshold,
        reward_exponent=args.reward_exponent,
        state_radius=args.state_radius,
    )
    dot_brackets = parse_dot_brackets(
        dataset=args.dataset,
        data_dir=args.data_dir,
        target_structure_ids=args.target_structure_ids,
    )
    learn_to_design_rna(
        dot_brackets,
        timeout=args.timeout,
        worker_count=args.worker_count or multiprocessing.cpu_count(),
        save_path=args.save_path,
        restore_path=args.restore_path,
        network_config=network_config,
        agent_config=agent_config,
        env_config=env_config,
    )
