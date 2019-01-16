import time

from pathlib import Path
from itertools import product
from dataclasses import dataclass
from distance import hamming

import numpy as np
from tensorforce.environments import Environment

from RNA import fold


@dataclass
class RnaDesignEnvironmentConfig:
    """
    Dataclass for the configuration of the environment.

    Default values describe:
        mutation_threshold: Defines the minimum distance needed before applying the local
            improvement step.
        reward_exponent: A parameter to shape the reward function.
        state_radius: The state representation is a (2*<state_radius> + 1)-gram
            at each position.
        use_conv: Bool to state if a convolutional network is used or not.
        use_embedding: Bool to state if embedding is used or not.
    """

    mutation_threshold: int = 5
    reward_exponent: float = 1.0
    state_radius: int = 5
    use_conv: bool = True
    use_embedding: bool = False


def _string_difference_indices(s1, s2):
    """
    Returns all indices where s1 and s2 differ.

    Args:
        s1: The first sequence.
        s2: The second sequence.

    Returns:
        List of indices where s1 and s2 differ.
    """
    return [index for index in range(len(s1)) if s1[index] != s2[index]]


def _encode_dot_bracket(secondary, environment_config):
    """
    Encode the dot_bracket notated target structure. The encoding can either be binary
    or by the embedding layer.

    Args:
        secondary: The target structure in dot_bracket notation.
        environment_config: The configuration of the environment.

    Returns:
        List of encoding for each site of the padded target structure.
    """
    padding = "=" * environment_config.state_radius
    padded_secondary = padding + secondary + padding

    if environment_config.use_embedding:
        site_encoding = {".": 0, "(": 1, ")": 2, "=": 3}
    else:
        site_encoding = {".": 0, "(": 1, ")": 1, "=": 0}

    # Sites corresponds to 1 pixel with 1 channel if convs are applied directly
    if environment_config.use_conv and not environment_config.use_embedding:
        return [[site_encoding[site]] for site in padded_secondary]
    return [site_encoding[site] for site in padded_secondary]


def _get_index_diff(string):
    """
    Compute the index of a corresponding closing bracket to an opening bracket.

    Args:
        string: The string to compute the index of corresponding closing bracket for.

    Returns:
        The index of the corresponding closing bracket for the first opening bracket
    """
    bracket_count = 0
    for index, base in enumerate(string):
        if base == "(":
            bracket_count += 1
        elif base == ")":
            bracket_count -= 1
        else:
            continue
        if bracket_count == 0:
            return index


def _encode_gap(secondary):
    """
    Encode the target structure using gap encoding.

    Args:
        secondary: The secondary structure that needs to get gap encoded.

    Returns:
        Gap encoding of the target secondary structure.

    Note:
        There exists a O(n) implementation of this algorithm using a stack.
    """
    gap_encoding = [0 for x in secondary]
    for index, base in enumerate(secondary):
        if base != "(":
            continue
        else:
            index_diff = _get_index_diff(secondary[index:])
            gap_encoding[index] = index_diff
            gap_encoding[index + index_diff] = index_diff * -1
    return gap_encoding


class _Target(object):
    """
    Class of the target structure. Provides encodings and id.
    """

    def __init__(self, dot_bracket, environment_config, id=None):
        """
        Initialize a target structure.

        Args:
             dot_bracket: dot_bracket encoded target structure.
             environment_config: The environment configuration.
             id: The id of the target structure.
        """
        self.id = id
        self.dot_bracket = dot_bracket
        self._gap_encoding = _encode_gap(self.dot_bracket)
        self.padded_encoding = _encode_dot_bracket(self.dot_bracket, environment_config)

    def __len__(self):
        return len(self.dot_bracket)

    def get_paired_site(self, site):
        """
        Get the paired site for <site> (base pair).

        Args:
            site: The site to check the pairing site for.

        Returns:
            The site that pairs with <site> if exists.
        """
        if self.dot_bracket[site] == ".":  # Has no paired site
            return None
        return self._gap_encoding[site] + site


class _Design(object):
    """
    Class of the designed candidate solution.
    """

    action_to_base = {0: "G", 1: "A", 2: "U", 3: "C"}
    action_to_pair = {0: "GC", 1: "CG", 2: "AU", 3: "UA"}

    def __init__(self, length=None, primary=None):
        """
        Initialize a candidate solution.

        Args:
            length: The length of the candidate solution.
            primary: The sequence of the candidate solution.
        """
        if primary:
            self._primary_list = primary
        else:
            self._primary_list = [None] * length
        self._dot_bracket = None

    def get_mutated(self, mutations, sites):
        """
        Locally change the candidate solution.

        Args:
            mutations: Possible mutations for the specified sites
            sites: The sites to be mutated

        Returns:
            A Design object with the mutated candidate solution.
        """
        mutatedprimary = self._primary_list.copy()
        for site, mutation in zip(sites, mutations):
            mutatedprimary[site] = mutation
        return _Design(primary=mutatedprimary)

    def assign_sites(self, action, site, paired_site=None):
        """
        Assign nucleotides to sites for designing a candidate solution.

        Args:
            action: The agents action to assign a nucleotide.
            site: The site to which the nucleotide is assigned to.
            paired_site: defines if the site is assigned with a base pair or not.
        """
        if paired_site:
            base_current, base_paired = self.action_to_pair[action]
            self._primary_list[site] = base_current
            self._primary_list[paired_site] = base_paired
        else:
            self._primary_list[site] = self.action_to_base[action]

    @property
    def primary(self):
        return "".join(self._primary_list)

    @property
    def dot_bracket(self):
        if not self._dot_bracket:
            self._dot_bracket, _ = fold(self.primary)
        return self._dot_bracket


def _random_epoch_gen(data):
    """
    Generator to get epoch data.

    Args:
        data: The targets of the epoch
    """
    while True:
        for i in np.random.permutation(len(data)):
            yield data[i]


def _get_target_gen(dataset, data_dir, target_ids, target_path, environment_config):
    """
    Generate the targets for next epoch.

    Args:
        dataset: The name of the benchmark to use targets from.
        data_dir: The directory of the target structures.
        target_ids: Use specific targets by ids.
        target path: Specify a path to the targets.
        environment_config: The configuration of the environment.

    Returns:
        An epoch generator for the specified target structures.
    """

    def _get_data_id_from_path(path):
        return int(path.name[:-4])

    if target_path:
        target_paths = [target_path]
    elif target_ids:
        target_paths = [Path(data_dir, dataset, f"{id_}.rna") for id_ in target_ids]
    else:
        target_paths = list(Path(data_dir, dataset).glob("*.rna"))
    target_ids = [_get_data_id_from_path(path) for path in target_paths]
    dot_brackets = [data_path.read_text().rstrip() for data_path in target_paths]
    targets = [
        _Target(dot_bracket, environment_config, id_)
        for dot_bracket, id_ in zip(dot_brackets, target_ids)
    ]
    return _random_epoch_gen(targets)


@dataclass
class EpisodeInfo:
    """
    Information class.
    """

    __slots__ = ["target_id", "time", "fractional_hamming_distance"]
    target_id: int
    time: float
    fractional_hamming_distance: float


class RnaDesignEnvironment(Environment):
    """
    The environment for RNA design using deep reinforcement learning.
    """

    def __init__(
        self,
        dataset,
        data_dir,
        target_structure_ids,
        target_structure_path,
        environment_config,
    ):
        """
        Initialize an environemnt.

        Args:
            dataset: The benchmark to use target structures from.
            data_dir: The path to the directory with all target structures.
            target_structure_ids: The ids of the target structures to use (optional).
            target_structure_path: The path to a target structure to use (optional).
            environment_config: The configuration of the environment.
        """
        self._mutation_threshold = environment_config.mutation_threshold
        self._reward_exponent = environment_config.reward_exponent
        self._state_radius = environment_config.state_radius
        self._use_embedding = environment_config.use_embedding
        self._use_conv = environment_config.use_conv

        self._target_gen = _get_target_gen(
            dataset,
            data_dir,
            target_structure_ids,
            target_structure_path,
            environment_config,
        )
        self.target = None
        self.design = None
        self._step = None
        self.episodes_info = []

    def __str__(self):
        return "RnaDesignEnvironment"

    def seed(self, seed):
        return None

    def reset(self):
        """
        Reset the environment. First function called by runner. Returns first state.

        Returns:
            The first state.
        """
        self.target = next(self._target_gen)
        self.design = _Design(len(self.target))
        self._step = 0
        return self._get_state()

    def _local_improvement(self):
        """
        Compute Hamming distance of locally improved candidate solutions.

        Returns:
            The minimum Hamming distance of all imporved candidate solutions.
        """
        differing_sites = _string_difference_indices(
            self.target.dot_bracket, self.design.dot_bracket
        )
        hamming_distances = []
        for mutation in product("AGCU", repeat=len(differing_sites)):
            mutated = self.design.get_mutated(mutation, differing_sites)
            hamming_distance = hamming(mutated.dot_bracket, self.target.dot_bracket)
            hamming_distances.append(hamming_distance)
            if hamming_distance == 0:  # For better timing results
                return 0
        return min(hamming_distances)

    def _apply_action(self, action):
        """
        Assign a nucleotide to a site.

        Args:
            action: The action chosen by the agent.
        """
        paired_site = self.target.get_paired_site(self._step)
        if paired_site:
            self.design.assign_sites(action, self._step, paired_site)
        else:
            self.design.assign_sites(action, self._step)

    def _get_state(self):
        """
        Get a state dependend on the padded encoding of the target structure.

        Returns:
            The next state.
        """
        return self.target.padded_encoding[
            self._step : self._step + 2 * self._state_radius + 1
        ]

    def _get_next_reward(self, terminal):
        """
        Compute the reward after assignment of all nucleotides.

        Args:
            terminal: Bool defining if final timestep is reached yet.

        Returns:
            The reward at the terminal timestep or 0 if not at the terminal timestep.
        """
        if not terminal:
            return 0

        hamming_distance = hamming(self.design.dot_bracket, self.target.dot_bracket)
        apply_mutation = (
            self._mutation_threshold and 0 < hamming_distance < self._mutation_threshold
        )
        if apply_mutation:
            locally_improved_distance = self._local_improvement()
            # If mutation found a solution use result
            if locally_improved_distance == 0:
                hamming_distance = locally_improved_distance

        fractional_hamming_distance = hamming_distance / len(self.target)

        # For hparam optimization
        episode_info = EpisodeInfo(
            target_id=self.target.id,
            time=time.time(),
            fractional_hamming_distance=fractional_hamming_distance,
        )
        self.episodes_info.append(episode_info)

        return (1 - fractional_hamming_distance) ** self._reward_exponent

    def execute(self, actions):
        """
        Execute one interaction of the environment with the agent.

        Args:
            action: Current action of the agent.

        Returns:
            state: The next state for the agent.
            terminal: The signal for end of an episode.
            reward: The reward if at terminal timestep, else 0.
        """
        self._apply_action(actions)
        self._step += 1

        terminal = self._step == len(self.target)
        state = None if terminal else self._get_state()
        reward = self._get_next_reward(terminal)

        return state, terminal, reward

    def close(self):
        pass

    @property
    def states(self):
        type = "int" if self._use_embedding else "float"
        if self._use_conv and not self._use_embedding:
            return dict(type=type, shape=(1 + 2 * self._state_radius, 1))
        return dict(type=type, shape=(1 + 2 * self._state_radius,))

    @property
    def actions(self):
        return dict(type="int", num_actions=4)
