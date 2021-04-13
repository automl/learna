import os
import shutil
import multiprocessing


import numpy as np
import ConfigSpace as CS
from hpbandster.core.worker import Worker

from learna.learna.agent import NetworkConfig, get_network, AgentConfig
from learna.learna.environment import RnaDesignEnvironment, RnaDesignEnvironmentConfig
from learna.learna.design_rna import design_rna
from learna.learna.learn_to_design_rna import learn_to_design_rna
from learna.data.parse_dot_brackets import parse_dot_brackets


class MetaLearnaWorker(Worker):
    def __init__(
        self, data_dir, num_cores, train_sequences, validation_timeout=60, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_cores = num_cores
        self.validation_timeout = validation_timeout
        self.train_sequences = parse_dot_brackets(
            dataset="rfam_learn/train",
            data_dir=data_dir,
            target_structure_ids=train_sequences,
        )
        self.validation_sequences = parse_dot_brackets(
            dataset="rfam_learn/validation",
            data_dir=data_dir,
            target_structure_ids=range(1, 101),
        )

    def compute(self, config, budget, working_directory, config_id, **kwargs):
        """
		Parameters
		----------
			budget: float
				cutoff for the agent on a single sequence
		"""

        tmp_dir = os.path.join(
            working_directory, "%i_%i_%i" % (config_id[0], config_id[1], config_id[2])
        )
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
        config = self._fill_config(config)

        network_config = NetworkConfig(
            conv_sizes=[config["conv_size1"], config["conv_size2"]],
            conv_channels=[config["conv_channels1"], config["conv_channels2"]],
            num_fc_layers=config["num_fc_layers"],
            fc_units=config["fc_units"],
            num_lstm_layers=config["num_lstm_layers"],
            lstm_units=config["lstm_units"],
            embedding_size=config["embedding_size"],
        )

        agent_config = AgentConfig(
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            entropy_regularization=config["entropy_regularization"],
        )

        env_config = RnaDesignEnvironmentConfig(
            reward_exponent=config["reward_exponent"], state_radius=config["state_radius"]
        )

        try:

            train_info = self._train(
                network_config, agent_config, env_config, tmp_dir, budget
            )
            validation_info = self._validate(
                network_config,
                agent_config,
                env_config,
                tmp_dir,
                config["restart_timeout"],
            )

        except:
            raise

        return {
            "loss": validation_info["sum_of_min_distances"],
            "info": {"train_info": train_info, "validation_info": validation_info},
        }

    def _train(self, network_config, agent_config, env_config, tmp_dir, budget):

        # create arguments for all sequences
        train_arguments = [
            self.train_sequences,
            budget,  # timeout
            self.num_cores,  # worker_count
            tmp_dir,  # save_path
            None,  # restore_path
            network_config,
            agent_config,
            env_config,
        ]
        # need to run tensoflow in a separate thread otherwise the pool in _evaluate
        # does not work
        with multiprocessing.Pool(1) as pool:
            train_results = pool.apply(learn_to_design_rna, train_arguments)

        train_results = process_train_results(train_results)

        train_sequence_infos = {}
        train_sum_of_min_distances = 0
        train_sum_of_last_distances = 0
        train_num_solved = 0

        for r in train_results.values():
            sequence_id = r[0].target_id
            r.sort(key=lambda e: e.time)

            dists = np.array(list(map(lambda e: e.normalized_hamming_distance, r)))

            train_sum_of_min_distances += dists.min()
            train_sum_of_last_distances += dists[-1]

            train_num_solved += dists.min() == 0.0

            train_sequence_infos[sequence_id] = {
                "num_episodes": len(r),
                "min_distance": float(dists.min()),
                "last_distance": float(dists[-1]),
            }

        train_info = {
            "num_solved": int(train_num_solved),
            "sum_of_min_distances": float(train_sum_of_min_distances),
            "sum_of_last_distances": float(train_sum_of_last_distances),
            "squence_infos": train_sequence_infos,
        }

        return train_info

    def _validate(
        self,
        network_config,
        agent_config,
        env_config,
        tmp_dir,
        restart_timeout,
        stop_learning=True,
    ):
        print("evaluating test performance")
        evaluation_arguments = [
            [
                [validation_sequence],
                self.validation_timeout,  # timeout
                tmp_dir,  # restore_path
                stop_learning,  # stop_learning
                restart_timeout,  # restart_timeout
                network_config,
                agent_config,
                env_config,
            ]
            for validation_sequence in self.validation_sequences
        ]

        with multiprocessing.Pool(self.num_cores) as pool:
            evaluation_results = pool.starmap(design_rna, evaluation_arguments)

        evaluation_sequence_infos = {}
        evaluation_sum_of_min_distances = 0
        evaluation_sum_of_first_distances = 0
        evaluation_num_solved = 0

        for r in evaluation_results:
            sequence_id = r[0].target_id
            r.sort(key=lambda e: e.time)

            times = np.array(list(map(lambda e: e.time, r)))
            dists = np.array(list(map(lambda e: e.normalized_hamming_distance, r)))

            evaluation_sum_of_min_distances += dists.min()
            evaluation_sum_of_first_distances += dists[0]

            evaluation_num_solved += dists.min() == 0.0

            evaluation_sequence_infos[sequence_id] = {
                "num_episodes": len(r),
                "mean_time_per_episode": float((times[1:] - times[:-1]).mean()),
                "min_distance": float(dists.min()),
                "last_distance": float(dists[-1]),
            }

        evaluation_info = {
            "num_solved": int(evaluation_num_solved),
            "sum_of_min_distances": float(evaluation_sum_of_min_distances),
            "sum_of_first_distances": float(evaluation_sum_of_first_distances),
            "squence_infos": evaluation_sequence_infos,
        }

        return evaluation_info

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()

        # parameters for PPO here
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                "learning_rate", lower=1e-6, upper=1e-4, log=True, default_value=1e-5
            )
        )
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "batch_size", lower=32, upper=128, log=True, default_value=32
            )
        )
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                "entropy_regularization",
                lower=5e-5,
                upper=5e-3,
                log=True,
                default_value=1.5e-3,
            )
        )
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                "reward_exponent", lower=1, upper=10, default_value=1
            )
        )
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                "state_radius_relative", lower=0, upper=1, default_value=0
            )
        )

        # parameters for the architecture
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "conv_radius1", lower=0, upper=8, default_value=1
            )
        )
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "conv_channels1", lower=1, upper=32, log=False, default_value=16
            )
        )

        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "conv_radius2", lower=0, upper=4, default_value=0
            )
        )
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "conv_channels2", lower=1, upper=32, log=False, default_value=1
            )
        )

        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "num_fc_layers", lower=1, upper=2, default_value=1
            )
        )
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "fc_units", lower=8, upper=64, log=True, default_value=50
            )
        )

        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "num_lstm_layers", lower=0, upper=2, default_value=0
            )
        )
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "lstm_units", lower=1, upper=64, log=True, default_value=1
            )
        )

        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "embedding_size", lower=0, upper=4, default_value=1
            )
        )

        return config_space

    @staticmethod
    def _fill_config(config):
        config["conv_size1"] = 1 + 2 * config["conv_radius1"]
        if config["conv_radius1"] == 0:
            config["conv_size1"] = 0
        del config["conv_radius1"]

        config["conv_size2"] = 1 + 2 * config["conv_radius2"]
        if config["conv_radius2"] == 0:
            config["conv_size2"] = 0
        del config["conv_radius2"]

        if config["conv_size1"] != 0:
            min_state_radius = config["conv_size1"] + config["conv_size1"] - 1
            max_state_radius = 32
            config["state_radius"] = int(
                min_state_radius
                + (max_state_radius - min_state_radius) * config["state_radius_relative"]
            )
            del config["state_radius_relative"]
        else:
            min_state_radius = config["conv_size2"] + config["conv_size2"] - 1
            max_state_radius = 32
            config["state_radius"] = int(
                min_state_radius
                + (max_state_radius - min_state_radius) * config["state_radius_relative"]
            )
            del config["state_radius_relative"]

        config["restart_timeout"] = None

        return config


def process_train_results(train_results):
    results_by_sequence = {}
    for r in train_results:
        for s in r:
            if not s.target_id in results_by_sequence:
                results_by_sequence[s.target_id] = [s]
            else:
                results_by_sequence[s.target_id].append(s)

    return results_by_sequence
