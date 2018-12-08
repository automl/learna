"""
    Testsuite for RNA-Design environment.
"""

from .environment import _get_index_diff
from .environment import _encode_gap
from .environment import RnaDesignEnvironment
from .environment import RnaDesignEnvironmentConfig
from .environment import _Target
from .environment import _Design
from .environment import _string_difference_indices
from .environment import _encode_dot_bracket
import pytest
import numpy.testing as nt
from distance import hamming
import numpy as np
from itertools import product


def test_get_index_diff():
    # test general behaviour
    string = "..(((.....))).."
    diff = _get_index_diff(string)
    nt.assert_equal(12, diff)

    # test empty string
    string = ""
    diff = _get_index_diff(string)
    nt.assert_equal(None, diff)

    # test brackets only
    string = "((()))"
    diff = _get_index_diff(string)
    nt.assert_equal(5, diff)

    # test without brackets
    string = "......."
    diff = _get_index_diff(string)
    nt.assert_equal(None, diff)


def test_encode_gap():
    secondary = "...(((.....))).."
    gap_encoding = [0, 0, 0, 10, 8, 6, 0, 0, 0, 0, 0, -6, -8, -10, 0, 0]
    test_gap = _encode_gap(secondary)
    nt.assert_equal(gap_encoding, test_gap)

    secondary = ""
    gap_encoding = []
    test_gap = _encode_gap(secondary)
    nt.assert_equal(gap_encoding, test_gap)

    secondary = "......"
    gap_encoding = [0, 0, 0, 0, 0, 0]
    test_gap = _encode_gap(secondary)
    nt.assert_equal(gap_encoding, test_gap)

    secondary = "(((())))"
    gap_encoding = [7, 5, 3, 1, -1, -3, -5, -7]
    test_gap = _encode_gap(secondary)
    nt.assert_equal(gap_encoding, test_gap)

    secondary = "..(((()).."
    with pytest.raises(Exception):
        test_gap = _encode_gap(secondary)


def test_string_difference_indices():
    s1 = "AAAbAAA"
    s2 = "AAAAAAA"
    nt.assert_equal([3], _string_difference_indices(s1, s2))
    s1 = "...((.......)).."
    s2 = "...(((.....))).."
    nt.assert_equal([5, 11], _string_difference_indices(s1, s2))


def test_encode_dot_bracket():
    secondary = "...((...)).."

    # no embedding, no convolution
    environment_config = RnaDesignEnvironmentConfig(state_radius=0)

    encoding = _encode_dot_bracket(secondary, environment_config)
    expected_encoding = [[0], [0], [0], [1], [1], [0], [0], [0], [1], [1], [0], [0]]

    nt.assert_equal(encoding, expected_encoding)

    environment_config = RnaDesignEnvironmentConfig(state_radius=3)

    encoding = _encode_dot_bracket(secondary, environment_config)
    expected_encoding = [
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [1],
        [1],
        [0],
        [0],
        [0],
        [1],
        [1],
        [0],
        [0],
        [0],
        [0],
        [0],
    ]

    nt.assert_equal(encoding, expected_encoding)

    # use embedding, no convolution
    environment_config = RnaDesignEnvironmentConfig(state_radius=3, use_embedding=True)

    encoding = _encode_dot_bracket(secondary, environment_config)
    expected_encoding = [3, 3, 3, 0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 0, 0, 3, 3, 3]

    nt.assert_equal(encoding, expected_encoding)

    environment_config = RnaDesignEnvironmentConfig(state_radius=0, use_embedding=True)

    encoding = _encode_dot_bracket(secondary, environment_config)
    expected_encoding = [0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 0, 0]

    nt.assert_equal(encoding, expected_encoding)

    # use convolution, no embedding
    environment_config = RnaDesignEnvironmentConfig(state_radius=0, use_conv=True)

    encoding = _encode_dot_bracket(secondary, environment_config)
    expected_encoding = [[0], [0], [0], [1], [1], [0], [0], [0], [1], [1], [0], [0]]

    nt.assert_equal(encoding, expected_encoding)

    environment_config = RnaDesignEnvironmentConfig(state_radius=3, use_conv=True)

    encoding = _encode_dot_bracket(secondary, environment_config)
    expected_encoding = [
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [1],
        [1],
        [0],
        [0],
        [0],
        [1],
        [1],
        [0],
        [0],
        [0],
        [0],
        [0],
    ]

    nt.assert_equal(encoding, expected_encoding)

    # use embedding, use convolution
    environment_config = RnaDesignEnvironmentConfig(
        state_radius=3, use_embedding=True, use_conv=True
    )

    encoding = _encode_dot_bracket(secondary, environment_config)
    expected_encoding = [3, 3, 3, 0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 0, 0, 3, 3, 3]

    nt.assert_equal(encoding, expected_encoding)

    environment_config = RnaDesignEnvironmentConfig(
        state_radius=0, use_embedding=True, use_conv=True
    )

    encoding = _encode_dot_bracket(secondary, environment_config)
    expected_encoding = [0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 0, 0]

    nt.assert_equal(encoding, expected_encoding)


def test_get_mutated():
    environment_config = RnaDesignEnvironmentConfig(include_mutation=True, state_radius=0)
    env = RnaDesignEnvironment(
        dataset="eterna",
        data_dir="data",
        target_structure_ids=None,
        target_structure_path=None,
        environment_config=environment_config,
    )
    env.target = _Target("...(((.....)))..", environment_config)

    env.design = _Design(
        primary=[
            "A",
            "A",
            "C",
            "G",
            "C",
            "A",
            "C",
            "G",
            "C",
            "C",
            "U",
            "G",
            "G",
            "C",
            "A",
            "C",
        ]
    )
    differing_sites = _string_difference_indices(
        env.target.dot_bracket, env.design.dot_bracket
    )
    distance = []
    for mutation in product("AGCU", repeat=len(differing_sites)):
        mutated = env.design.get_mutated(mutation, differing_sites)
        distance.append(hamming(env.target.dot_bracket, mutated.dot_bracket))

    assert 0 in distance


def test_assign_sites():
    actions = [0, 1, 0, 1, 2, 1]
    paired_site = [7, 6, None, None, None, None]
    sites = [0, 1, 2, 3, 4, 5]

    design = _Design(length=8)

    for index, action in enumerate(actions):
        design.assign_sites(action, sites[index], paired_site[index])

    test_primary = "GCGAUAGC"
    test_secondary = "((....))"
    assert design.primary == test_primary
    assert design.dot_bracket == test_secondary


def test_locally_improved_distance():
    environment_config = RnaDesignEnvironmentConfig(
        mutation_threshold=5,
        include_mutation=True,
        reward_exponent=1,
        state_radius=0,
        use_conv=False,
    )
    env = RnaDesignEnvironment(
        dataset="eterna",
        data_dir="data",
        target_structure_ids=None,
        target_structure_path=None,
        environment_config=environment_config,
    )
    env.target = _Target("...(((.....)))..", environment_config)

    env.design = _Design(
        primary=[
            "A",
            "A",
            "C",
            "G",
            "C",
            "A",
            "C",
            "G",
            "C",
            "C",
            "U",
            "G",
            "G",
            "C",
            "A",
            "C",
        ]
    )
    locally_improved_distance = env._local_improvement()
    assert locally_improved_distance in [0, 2]

    assert env.design.dot_bracket != env.target.dot_bracket


def test_general_binary_encoding():
    # setup sequences
    designed_sequence = "AAAGUAAAAAAUACAA"
    target_sequence = "AAAGGGAAAAACCCAA"
    designed_dot_bracket = "................"
    target_dot_bracket = "...(((.....))).."
    fractional_hamming_distance = hamming(target_dot_bracket, designed_dot_bracket) / len(
        target_dot_bracket
    )

    # setup data
    def _random_epoch_gen(data):
        while True:
            yield data[0]

    # setup environment
    environment_config = RnaDesignEnvironmentConfig(
        mutation_threshold=7,
        include_mutation=False,
        reward_exponent=1,
        state_radius=0,
        use_conv=False,
    )

    env = RnaDesignEnvironment(
        dataset="eterna",
        data_dir="data",
        target_structure_ids=None,
        target_structure_path=None,
        environment_config=environment_config,
    )

    env._mutation_threshold = 0
    env._include_mutation = False
    env._reward_exponent = 1
    env._state_radius = 0
    env._use_conv = False

    target = _Target(target_dot_bracket, environment_config)
    env._target_gen = _random_epoch_gen([target])
    env.target = target
    env.design = _Design(
        primary=[
            "A",
            "A",
            "A",
            "G",
            "U",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "U",
            "A",
            "C",
            "A",
            "A",
        ]
    )
    env._step = None
    env.episodes_info = []

    # reset environment, get first state
    state = env.reset()
    assert state == [0]
    assert env._step == 0
    assert env.target.dot_bracket[env._step] == "."
    assert env.target.dot_bracket == target_dot_bracket

    # setup state, actions, terminal, reward, steps, sites for one run on target_dot_bracket
    states = [[0], [0], [1], [1], [1], [0], [0], [0], [0], [0], [1], [1], [1], [0], [0]]
    actions = [1, 1, 1, 0, 3, 2, 1, 1, 1, 1, 1, 3, 2, 1, 1]
    terminals = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    rewards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sites = "...(((.....)))."

    # run one episode until timestep t = T - 1
    for (
        action_desired,
        state_desired,
        terminal_desired,
        reward_desired,
        site_desired,
        step_desired,
    ) in zip(actions, states, terminals, rewards, sites, steps):

        state, terminal, reward = env.execute(action_desired)
        assert state == state_desired
        assert terminal == terminal_desired
        assert reward == reward_desired
        assert env._step == step_desired
        assert env.target.dot_bracket[env._step - 1] == site_desired
        assert env.target.dot_bracket == target_dot_bracket

    # run last timestep including reward
    state, terminal, reward = env.execute(1)
    assert state == None
    assert terminal == True
    assert reward == 1 - fractional_hamming_distance
    assert env.design.primary == designed_sequence
