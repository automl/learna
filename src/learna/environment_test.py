"""
    Testsuite for RNA-Design environment.
"""

import pytest
import numpy as np
import numpy.testing as nt

from distance import hamming
from itertools import product

from .environment import _string_difference_indices
from .environment import RnaDesignEnvironmentConfig
from .environment import _encode_dot_bracket
from .environment import _encode_pairing
from .environment import _Target
from .environment import _Design
from .environment import RnaDesignEnvironment

from RNA import fold


def test_string_difference_indices():
    # Test general behaviour
    s1 = "...(((..))).."
    s2 = "...((....)).."

    assert [5, 8] == _string_difference_indices(s1, s2)

    # Test equal strings
    s1 = "....."
    s2 = "....."

    assert [] == _string_difference_indices(s1, s2)

    # Test empty input s1
    s1 = ""
    s2 = "...().."

    assert [] == _string_difference_indices(s1, s2)

    # Test empty input s2
    s1 = "....."
    s2 = ""

    nt.assert_raises(
        IndexError, _string_difference_indices, s1, s2
    )  # TODO: Handle IndexError


def test_encode_dot_bracket():
    secondary = "..((..))."
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=False, state_radius=0
    )

    # Assert general behaviour
    assert [int(site) for site in "001100110"] == _encode_dot_bracket(
        secondary, environment_config
    )

    # Test padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=False, state_radius=1
    )
    assert [int(site) for site in "00011001100"] == _encode_dot_bracket(
        secondary, environment_config
    )

    # Test embedding encoding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=True, state_radius=0
    )
    assert [int(site) for site in "001100220"] == _encode_dot_bracket(
        secondary, environment_config
    )

    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=True, state_radius=1
    )
    assert [int(site) for site in "30011002203"] == _encode_dot_bracket(
        secondary, environment_config
    )

    # Test convolution encoding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=False, state_radius=0
    )
    assert [[int(site)] for site in "001100110"] == _encode_dot_bracket(
        secondary, environment_config
    )

    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=False, state_radius=1
    )
    assert [[int(site)] for site in "00011001100"] == _encode_dot_bracket(
        secondary, environment_config
    )

    # Ignore convolution encoding if embedding is used
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=True, state_radius=0
    )
    assert [int(site) for site in "001100220"] == _encode_dot_bracket(
        secondary, environment_config
    )

    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=True, state_radius=1
    )
    assert [int(site) for site in "30011002203"] == _encode_dot_bracket(
        secondary, environment_config
    )


def test_encode_pairing():
    secondary = "..((..))."

    # Test general behaviour
    assert [None, None, 7, 6, None, None, 3, 2, None] == _encode_pairing(secondary)

    # Test no pairs in sequence
    secondary = "....."
    assert [None for site in secondary] == _encode_pairing(secondary)

    # Test sequence of pairs only
    secondary = "((()))"
    assert [5, 4, 3, 2, 1, 0] == _encode_pairing(secondary)

    # Test empty input
    secondary = ""
    assert [] == _encode_pairing(secondary)


def test_Target():
    dot_bracket = "..((..))."

    # No conv, no embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=False, state_radius=0
    )

    target = _Target(dot_bracket, environment_config)
    assert 1 == target.id
    assert "..((..))." == target.dot_bracket
    assert [int(site) for site in "001100110"] == target.padded_encoding

    [
        nt.assert_equal(site, target.get_paired_site(index))
        for index, site in enumerate([None, None, 7, 6, None, None, 3, 2, None])
    ]

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=False, state_radius=1
    )

    target = _Target(dot_bracket, environment_config)
    assert 2 == target.id
    assert "..((..))." == target.dot_bracket
    assert [int(site) for site in "00011001100"] == target.padded_encoding

    [
        nt.assert_equal(site, target.get_paired_site(index))
        for index, site in enumerate([None, None, 7, 6, None, None, 3, 2, None])
    ]

    # No conv, embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=True, state_radius=0
    )

    target = _Target(dot_bracket, environment_config)
    assert 3 == target.id
    assert "..((..))." == target.dot_bracket
    assert [int(site) for site in "001100220"] == target.padded_encoding

    [
        nt.assert_equal(site, target.get_paired_site(index))
        for index, site in enumerate([None, None, 7, 6, None, None, 3, 2, None])
    ]

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=True, state_radius=1
    )

    target = _Target(dot_bracket, environment_config)
    assert 4 == target.id
    assert "..((..))." == target.dot_bracket
    assert [int(site) for site in "30011002203"] == target.padded_encoding

    [
        nt.assert_equal(site, target.get_paired_site(index))
        for index, site in enumerate([None, None, 7, 6, None, None, 3, 2, None])
    ]

    # Conv, embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=True, state_radius=0
    )

    target = _Target(dot_bracket, environment_config)
    assert 5 == target.id
    assert "..((..))." == target.dot_bracket
    assert [int(site) for site in "001100220"] == target.padded_encoding

    [
        nt.assert_equal(site, target.get_paired_site(index))
        for index, site in enumerate([None, None, 7, 6, None, None, 3, 2, None])
    ]

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=True, state_radius=1
    )

    target = _Target(dot_bracket, environment_config)
    assert 6 == target.id
    assert "..((..))." == target.dot_bracket
    assert [int(site) for site in "30011002203"] == target.padded_encoding

    [
        nt.assert_equal(site, target.get_paired_site(index))
        for index, site in enumerate([None, None, 7, 6, None, None, 3, 2, None])
    ]

    # Conv, no embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=False, state_radius=0
    )

    target = _Target(dot_bracket, environment_config)
    assert 7 == target.id
    assert "..((..))." == target.dot_bracket
    assert [[int(site)] for site in "001100110"] == target.padded_encoding

    [
        nt.assert_equal(site, target.get_paired_site(index))
        for index, site in enumerate([None, None, 7, 6, None, None, 3, 2, None])
    ]

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=False, state_radius=1
    )

    target = _Target(dot_bracket, environment_config)
    assert 8 == target.id
    assert "..((..))." == target.dot_bracket
    assert [[int(site)] for site in "00011001100"] == target.padded_encoding

    [
        nt.assert_equal(site, target.get_paired_site(index))
        for index, site in enumerate([None, None, 7, 6, None, None, 3, 2, None])
    ]


def test_Design():
    # Test Initialization
    with nt.assert_raises(TypeError):
        design = _Design()  # TODO: Handle TypeError caused by NoneType initialization
    dot_bracket = "..((..))."

    # No conv, no embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=False, state_radius=0
    )

    target = _Target(dot_bracket, environment_config)
    design = _Design(length=len(target))

    with nt.assert_raises(TypeError):
        design.primary  # TODO: Handle TypeError caused by NoneType init

    assert design.first_unassigned_site == 0

    # Test nucleotide assignment
    for i in range(len(dot_bracket)):
        if dot_bracket[i] == ")":
            continue
        action = 0
        site = design.first_unassigned_site
        design.assign_sites(action, site, paired_site=target.get_paired_site(site))

    assert "GGGGGGCCG" == design.primary

    # Test Local Improvement procedure
    mutations = "AGCU"

    for mutation in mutations:
        site = [1]
        mutated = design.get_mutated(mutation, site)
        assert f"G{mutation}GGGGCCG" == mutated.primary

        site = [2]
        mutated = design.get_mutated(mutation, site)
        assert f"GG{mutation}GGGCCG" == mutated.primary

        site = [3]
        mutated = design.get_mutated(mutation, site)
        assert f"GGG{mutation}GGCCG" == mutated.primary

        site = [4]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGG{mutation}GCCG" == mutated.primary

        site = [5]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGG{mutation}CCG" == mutated.primary

        site = [6]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGG{mutation}CG" == mutated.primary

        site = [7]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGGC{mutation}G" == mutated.primary

        site = [8]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGGCC{mutation}" == mutated.primary

        with nt.assert_raises(IndexError):
            site = [9]
            mutated = design.get_mutated(mutation, site)

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=False, state_radius=1
    )

    target = _Target(dot_bracket, environment_config)
    design = _Design(length=len(target))

    with nt.assert_raises(TypeError):
        design.primary  # TODO: Handle TypeError caused by NoneType init

    assert design.first_unassigned_site == 0

    # Test nucleotide assignment
    for i in range(len(dot_bracket)):
        if dot_bracket[i] == ")":
            continue
        action = 0
        site = design.first_unassigned_site
        design.assign_sites(action, site, paired_site=target.get_paired_site(site))

    assert "GGGGGGCCG" == design.primary

    # Test Local Improvement procedure
    mutations = "AGCU"

    for mutation in mutations:
        site = [1]
        mutated = design.get_mutated(mutation, site)
        assert f"G{mutation}GGGGCCG" == mutated.primary

        site = [2]
        mutated = design.get_mutated(mutation, site)
        assert f"GG{mutation}GGGCCG" == mutated.primary

        site = [3]
        mutated = design.get_mutated(mutation, site)
        assert f"GGG{mutation}GGCCG" == mutated.primary

        site = [4]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGG{mutation}GCCG" == mutated.primary

        site = [5]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGG{mutation}CCG" == mutated.primary

        site = [6]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGG{mutation}CG" == mutated.primary

        site = [7]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGGC{mutation}G" == mutated.primary

        site = [8]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGGCC{mutation}" == mutated.primary

        with nt.assert_raises(IndexError):
            site = [9]
            mutated = design.get_mutated(mutation, site)

    # No conv, embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=True, state_radius=0
    )

    target = _Target(dot_bracket, environment_config)
    design = _Design(length=len(target))

    with nt.assert_raises(TypeError):
        design.primary  # TODO: Handle TypeError caused by NoneType init

    assert design.first_unassigned_site == 0

    # Test nucleotide assignment
    for i in range(len(dot_bracket)):
        if dot_bracket[i] == ")":
            continue
        action = 0
        site = design.first_unassigned_site
        design.assign_sites(action, site, paired_site=target.get_paired_site(site))

    assert "GGGGGGCCG" == design.primary

    # Test Local Improvement procedure
    mutations = "AGCU"

    for mutation in mutations:
        site = [1]
        mutated = design.get_mutated(mutation, site)
        assert f"G{mutation}GGGGCCG" == mutated.primary

        site = [2]
        mutated = design.get_mutated(mutation, site)
        assert f"GG{mutation}GGGCCG" == mutated.primary

        site = [3]
        mutated = design.get_mutated(mutation, site)
        assert f"GGG{mutation}GGCCG" == mutated.primary

        site = [4]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGG{mutation}GCCG" == mutated.primary

        site = [5]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGG{mutation}CCG" == mutated.primary

        site = [6]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGG{mutation}CG" == mutated.primary

        site = [7]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGGC{mutation}G" == mutated.primary

        site = [8]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGGCC{mutation}" == mutated.primary

        with nt.assert_raises(IndexError):
            site = [9]
            mutated = design.get_mutated(mutation, site)

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=True, state_radius=1
    )

    target = _Target(dot_bracket, environment_config)
    design = _Design(length=len(target))

    with nt.assert_raises(TypeError):
        design.primary  # TODO: Handle TypeError caused by NoneType init

    assert design.first_unassigned_site == 0

    # Test nucleotide assignment
    for i in range(len(dot_bracket)):
        if dot_bracket[i] == ")":
            continue
        action = 0
        site = design.first_unassigned_site
        design.assign_sites(action, site, paired_site=target.get_paired_site(site))

    assert "GGGGGGCCG" == design.primary

    # Test Local Improvement procedure
    mutations = "AGCU"

    for mutation in mutations:
        site = [1]
        mutated = design.get_mutated(mutation, site)
        assert f"G{mutation}GGGGCCG" == mutated.primary

        site = [2]
        mutated = design.get_mutated(mutation, site)
        assert f"GG{mutation}GGGCCG" == mutated.primary

        site = [3]
        mutated = design.get_mutated(mutation, site)
        assert f"GGG{mutation}GGCCG" == mutated.primary

        site = [4]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGG{mutation}GCCG" == mutated.primary

        site = [5]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGG{mutation}CCG" == mutated.primary

        site = [6]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGG{mutation}CG" == mutated.primary

        site = [7]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGGC{mutation}G" == mutated.primary

        site = [8]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGGCC{mutation}" == mutated.primary

        with nt.assert_raises(IndexError):
            site = [9]
            mutated = design.get_mutated(mutation, site)

    # Conv, embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=True, state_radius=0
    )

    target = _Target(dot_bracket, environment_config)
    design = _Design(length=len(target))

    with nt.assert_raises(TypeError):
        design.primary  # TODO: Handle TypeError caused by NoneType init

    assert design.first_unassigned_site == 0

    # Test nucleotide assignment
    for i in range(len(dot_bracket)):
        if dot_bracket[i] == ")":
            continue
        action = 0
        site = design.first_unassigned_site
        design.assign_sites(action, site, paired_site=target.get_paired_site(site))

    assert "GGGGGGCCG" == design.primary

    # Test Local Improvement procedure
    mutations = "AGCU"

    for mutation in mutations:
        site = [1]
        mutated = design.get_mutated(mutation, site)
        assert f"G{mutation}GGGGCCG" == mutated.primary

        site = [2]
        mutated = design.get_mutated(mutation, site)
        assert f"GG{mutation}GGGCCG" == mutated.primary

        site = [3]
        mutated = design.get_mutated(mutation, site)
        assert f"GGG{mutation}GGCCG" == mutated.primary

        site = [4]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGG{mutation}GCCG" == mutated.primary

        site = [5]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGG{mutation}CCG" == mutated.primary

        site = [6]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGG{mutation}CG" == mutated.primary

        site = [7]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGGC{mutation}G" == mutated.primary

        site = [8]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGGCC{mutation}" == mutated.primary

        with nt.assert_raises(IndexError):
            site = [9]
            mutated = design.get_mutated(mutation, site)

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=True, state_radius=1
    )

    target = _Target(dot_bracket, environment_config)
    design = _Design(length=len(target))

    with nt.assert_raises(TypeError):
        design.primary  # TODO: Handle TypeError caused by NoneType init

    assert design.first_unassigned_site == 0

    # Test nucleotide assignment
    for i in range(len(dot_bracket)):
        if dot_bracket[i] == ")":
            continue
        action = 0
        site = design.first_unassigned_site
        design.assign_sites(action, site, paired_site=target.get_paired_site(site))

    assert "GGGGGGCCG" == design.primary

    # Test Local Improvement procedure
    mutations = "AGCU"

    for mutation in mutations:
        site = [1]
        mutated = design.get_mutated(mutation, site)
        assert f"G{mutation}GGGGCCG" == mutated.primary

        site = [2]
        mutated = design.get_mutated(mutation, site)
        assert f"GG{mutation}GGGCCG" == mutated.primary

        site = [3]
        mutated = design.get_mutated(mutation, site)
        assert f"GGG{mutation}GGCCG" == mutated.primary

        site = [4]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGG{mutation}GCCG" == mutated.primary

        site = [5]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGG{mutation}CCG" == mutated.primary

        site = [6]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGG{mutation}CG" == mutated.primary

        site = [7]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGGC{mutation}G" == mutated.primary

        site = [8]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGGCC{mutation}" == mutated.primary

        with nt.assert_raises(IndexError):
            site = [9]
            mutated = design.get_mutated(mutation, site)

    # Conv, no embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=False, state_radius=0
    )

    target = _Target(dot_bracket, environment_config)
    design = _Design(length=len(target))

    with nt.assert_raises(TypeError):
        design.primary  # TODO: Handle TypeError caused by NoneType init

    assert design.first_unassigned_site == 0

    # Test nucleotide assignment
    for i in range(len(dot_bracket)):
        if dot_bracket[i] == ")":
            continue
        action = 0
        site = design.first_unassigned_site
        design.assign_sites(action, site, paired_site=target.get_paired_site(site))

    assert "GGGGGGCCG" == design.primary

    # Test Local Improvement procedure
    mutations = "AGCU"

    for mutation in mutations:
        site = [1]
        mutated = design.get_mutated(mutation, site)
        assert f"G{mutation}GGGGCCG" == mutated.primary

        site = [2]
        mutated = design.get_mutated(mutation, site)
        assert f"GG{mutation}GGGCCG" == mutated.primary

        site = [3]
        mutated = design.get_mutated(mutation, site)
        assert f"GGG{mutation}GGCCG" == mutated.primary

        site = [4]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGG{mutation}GCCG" == mutated.primary

        site = [5]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGG{mutation}CCG" == mutated.primary

        site = [6]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGG{mutation}CG" == mutated.primary

        site = [7]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGGC{mutation}G" == mutated.primary

        site = [8]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGGCC{mutation}" == mutated.primary

        with nt.assert_raises(IndexError):
            site = [9]
            mutated = design.get_mutated(mutation, site)

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=False, state_radius=1
    )

    target = _Target(dot_bracket, environment_config)
    design = _Design(length=len(target))

    with nt.assert_raises(TypeError):
        design.primary  # TODO: Handle TypeError caused by NoneType init

    assert design.first_unassigned_site == 0

    # Test nucleotide assignment
    for i in range(len(dot_bracket)):
        if dot_bracket[i] == ")":
            continue
        action = 0
        site = design.first_unassigned_site
        design.assign_sites(action, site, paired_site=target.get_paired_site(site))

    assert "GGGGGGCCG" == design.primary

    # Test Local Improvement procedure
    mutations = "AGCU"

    for mutation in mutations:
        site = [1]
        mutated = design.get_mutated(mutation, site)
        assert f"G{mutation}GGGGCCG" == mutated.primary

        site = [2]
        mutated = design.get_mutated(mutation, site)
        assert f"GG{mutation}GGGCCG" == mutated.primary

        site = [3]
        mutated = design.get_mutated(mutation, site)
        assert f"GGG{mutation}GGCCG" == mutated.primary

        site = [4]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGG{mutation}GCCG" == mutated.primary

        site = [5]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGG{mutation}CCG" == mutated.primary

        site = [6]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGG{mutation}CG" == mutated.primary

        site = [7]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGGC{mutation}G" == mutated.primary

        site = [8]
        mutated = design.get_mutated(mutation, site)
        assert f"GGGGGGCC{mutation}" == mutated.primary

        with nt.assert_raises(IndexError):
            site = [9]
            mutated = design.get_mutated(mutation, site)


def test_RnaDesignEnvironment_reset():
    dot_brackets = ["..((..))."]

    # No conv, no embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=False, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    assert None == environment.target
    assert None == environment.design

    assert [0] == environment.reset()
    assert "..((..))." == environment.target.dot_bracket
    assert [int(site) for site in "001100110"] == environment.target.padded_encoding

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=False, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    assert None == environment.target
    assert None == environment.design

    assert [0, 0, 0] == environment.reset()
    assert "..((..))." == environment.target.dot_bracket
    assert [int(site) for site in "00011001100"] == environment.target.padded_encoding

    # No conv, embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=True, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    assert None == environment.target
    assert None == environment.design

    assert [0] == environment.reset()
    assert "..((..))." == environment.target.dot_bracket
    assert [int(site) for site in "001100220"] == environment.target.padded_encoding

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=True, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    assert None == environment.target
    assert None == environment.design

    assert [3, 0, 0] == environment.reset()
    assert "..((..))." == environment.target.dot_bracket
    assert [int(site) for site in "30011002203"] == environment.target.padded_encoding

    # Conv, no embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=False, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    assert None == environment.target
    assert None == environment.design

    assert [[0]] == environment.reset()
    assert "..((..))." == environment.target.dot_bracket
    assert [[int(site)] for site in "001100110"] == environment.target.padded_encoding

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=False, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    assert None == environment.target
    assert None == environment.design

    assert [[0], [0], [0]] == environment.reset()
    assert "..((..))." == environment.target.dot_bracket
    assert [[int(site)] for site in "00011001100"] == environment.target.padded_encoding

    # Conv, embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=True, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    assert None == environment.target
    assert None == environment.design

    assert [0] == environment.reset()
    assert "..((..))." == environment.target.dot_bracket
    assert [int(site) for site in "001100220"] == environment.target.padded_encoding

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=True, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    assert None == environment.target
    assert None == environment.design

    assert [3, 0, 0] == environment.reset()
    assert "..((..))." == environment.target.dot_bracket
    assert [int(site) for site in "30011002203"] == environment.target.padded_encoding


def test_RnaDesignEnvironment_apply_action():
    primary_list = list("GGGGGGCCG")
    dot_brackets = ["..((..))."]

    # No conv, no embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=True, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    environment.reset()

    i = 0

    while environment.design.first_unassigned_site is not None:
        action = 0
        environment._apply_action(action)
        assert environment.design._primary_list[:i] == primary_list[:i]
        i += 1
    assert "GGGGGGCCG" == environment.design.primary

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=True, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    environment.reset()

    i = 0

    while environment.design.first_unassigned_site is not None:
        action = 0
        environment._apply_action(action)
        assert environment.design._primary_list[:i] == primary_list[:i]
        i += 1
    assert "GGGGGGCCG" == environment.design.primary

    # No conv, embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=False, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    environment.reset()

    i = 0

    while environment.design.first_unassigned_site is not None:
        action = 0
        environment._apply_action(action)
        assert environment.design._primary_list[:i] == primary_list[:i]
        i += 1
    assert "GGGGGGCCG" == environment.design.primary

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=False, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    environment.reset()

    i = 0

    while environment.design.first_unassigned_site is not None:
        action = 0
        environment._apply_action(action)
        assert environment.design._primary_list[:i] == primary_list[:i]
        i += 1
    assert "GGGGGGCCG" == environment.design.primary

    # Conv, no embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=False, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    environment.reset()

    i = 0

    while environment.design.first_unassigned_site is not None:
        action = 0
        environment._apply_action(action)
        assert environment.design._primary_list[:i] == primary_list[:i]
        i += 1
    assert "GGGGGGCCG" == environment.design.primary

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=False, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    environment.reset()

    i = 0

    while environment.design.first_unassigned_site is not None:
        action = 0
        environment._apply_action(action)
        assert environment.design._primary_list[:i] == primary_list[:i]
        i += 1
    assert "GGGGGGCCG" == environment.design.primary

    # Conv, embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=True, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    environment.reset()

    i = 0

    while environment.design.first_unassigned_site is not None:
        action = 0
        environment._apply_action(action)
        assert environment.design._primary_list[:i] == primary_list[:i]
        i += 1
    assert "GGGGGGCCG" == environment.design.primary

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=True, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    environment.reset()

    i = 0

    while environment.design.first_unassigned_site is not None:
        action = 0
        environment._apply_action(action)
        assert environment.design._primary_list[:i] == primary_list[:i]
        i += 1
    assert "GGGGGGCCG" == environment.design.primary


def test_RnaDesignEnvironment_get_state():
    dot_brackets = ["..((..))."]

    # No conv, no embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=False, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    environment.reset()

    states = [[state] for state in environment.target.padded_encoding]

    while environment.design.first_unassigned_site is not None:
        assert (
            states[environment.design.first_unassigned_site] == environment._get_state()
        )
        environment._apply_action(0)

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=False, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    environment.reset()

    states = [
        [
            environment.target.padded_encoding[i - 1],
            environment.target.padded_encoding[i],
            environment.target.padded_encoding[i + 1],
        ]
        for i in range(len(environment.target.padded_encoding) - 1)
        if i > 0
    ]

    while environment.design.first_unassigned_site is not None:
        assert (
            states[environment.design.first_unassigned_site] == environment._get_state()
        )
        environment._apply_action(0)

    # No conv, embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=True, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    environment.reset()

    states = [[state] for state in environment.target.padded_encoding]

    while environment.design.first_unassigned_site is not None:
        assert (
            states[environment.design.first_unassigned_site] == environment._get_state()
        )
        environment._apply_action(0)

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=True, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    environment.reset()

    states = [
        [
            environment.target.padded_encoding[i - 1],
            environment.target.padded_encoding[i],
            environment.target.padded_encoding[i + 1],
        ]
        for i in range(len(environment.target.padded_encoding) - 1)
        if i > 0
    ]

    while environment.design.first_unassigned_site is not None:
        assert (
            states[environment.design.first_unassigned_site] == environment._get_state()
        )
        environment._apply_action(0)

    # Conv, no embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=False, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    environment.reset()

    states = [[state] for state in environment.target.padded_encoding]

    while environment.design.first_unassigned_site is not None:
        assert (
            states[environment.design.first_unassigned_site] == environment._get_state()
        )
        environment._apply_action(0)

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=False, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    environment.reset()

    states = [
        [
            environment.target.padded_encoding[i - 1],
            environment.target.padded_encoding[i],
            environment.target.padded_encoding[i + 1],
        ]
        for i in range(len(environment.target.padded_encoding) - 1)
        if i > 0
    ]

    while environment.design.first_unassigned_site is not None:
        assert (
            states[environment.design.first_unassigned_site] == environment._get_state()
        )
        environment._apply_action(0)

    # Conv, embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=True, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    environment.reset()

    states = [[state] for state in environment.target.padded_encoding]

    while environment.design.first_unassigned_site is not None:
        assert (
            states[environment.design.first_unassigned_site] == environment._get_state()
        )
        environment._apply_action(0)

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=True, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    environment.reset()

    states = [
        [
            environment.target.padded_encoding[i - 1],
            environment.target.padded_encoding[i],
            environment.target.padded_encoding[i + 1],
        ]
        for i in range(len(environment.target.padded_encoding) - 1)
        if i > 0
    ]

    while environment.design.first_unassigned_site is not None:
        assert (
            states[environment.design.first_unassigned_site] == environment._get_state()
        )
        environment._apply_action(0)


# TODO: Find example with final distance > 0 to test for distance list
def test_RnaDesignEnvironment_local_improvement():
    dot_brackets = ["((....))"]

    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=True, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    environment.reset()

    # Assign nucleotides such that design.primary folds correctly
    # Possible solution: GCGAUAGC
    environment._apply_action(0)
    environment._apply_action(1)
    environment._apply_action(0)
    environment._apply_action(1)
    environment._apply_action(2)
    environment._apply_action(1)

    assert "GCGAUAGC" == environment.design.primary

    # NOTE: Hamming distances may differ if other folding algorithm or new version is used
    assert 0 == hamming(
        environment.target.dot_bracket, fold(environment.design.primary)[0]
    )

    # Mutate primary sequence before local improvement step
    mutated = environment.design.get_mutated("A", [0])
    assert "ACGAUAGC" == mutated.primary
    assert "........" == fold(mutated.primary)[0]
    hamming_distance = hamming(environment.target.dot_bracket, fold(mutated.primary)[0])
    assert 0 < hamming_distance
    assert 4 == hamming_distance
    assert 0 == environment._local_improvement(fold(mutated.primary)[0])

    mutated = environment.design.get_mutated("AA", [0, 1])
    assert "AAGAUAGC" == mutated.primary
    assert "........" == fold(mutated.primary)[0]
    hamming_distance = hamming(environment.target.dot_bracket, fold(mutated.primary)[0])
    assert 0 < hamming_distance
    assert 4 == hamming_distance
    assert 0 == environment._local_improvement(fold(mutated.primary)[0])

    mutated = environment.design.get_mutated("AAAAAAAA", [0, 1, 2, 3, 4, 5, 6, 7])
    assert "AAAAAAAA" == mutated.primary
    assert "........" == fold(mutated.primary)[0]
    hamming_distance = hamming(environment.target.dot_bracket, fold(mutated.primary)[0])
    assert 0 < hamming_distance
    assert 4 == hamming_distance
    assert 0 == environment._local_improvement(fold(mutated.primary)[0])


def test_RnaDesignEnvironment_get_reward():
    dot_brackets = ["(((....)))"]

    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=True, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)
    environment.reset()

    # Assign nucleotides such that design.primary folds correctly
    # Possible solution: CGCCUACGCG
    environment._apply_action(1)
    environment._apply_action(0)
    environment._apply_action(1)
    environment._apply_action(3)
    environment._apply_action(2)
    environment._apply_action(1)
    environment._apply_action(3)

    # Test general behaviour
    assert "CGCCUACGCG" == environment.design.primary
    assert 0 == environment._get_reward(False)
    assert 1.0 == environment._get_reward(True)

    # Mutate primary sequence to change secondary structure
    # NOTE: Hamming distance is larger than mutation_threshold (5), no local improvement is applied
    mutated = environment.design.get_mutated(
        tuple(list("AAAAAAAAAA")), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    assert "AAAAAAAAAA" == mutated.primary
    assert ".........." == fold(mutated.primary)[0]
    hamming_distance = hamming(environment.target.dot_bracket, fold(mutated.primary)[0])
    assert 0 < hamming_distance
    assert 6 == hamming_distance

    environment.design = mutated
    assert 1.0 > environment._get_reward(True)

    hamming_distance = hamming(
        environment.target.dot_bracket, fold(environment.design.primary)[0]
    )
    normalized_hamming_distance = (
        1
        - (hamming_distance / len(environment.target))
        ** environment._env_config.reward_exponent
    )
    assert normalized_hamming_distance == environment._get_reward(True)

    # Mutate primary sequence to change secondary structure
    # NOTE: Hamming distance is smaller than mutation_threshold (5), local improvement is applied
    mutated = environment.design.get_mutated(
        tuple(list("GGCCUAUGCG")), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    assert "GGCCUAUGCG" == mutated.primary
    assert ".((....))." == fold(mutated.primary)[0]
    hamming_distance = hamming(environment.target.dot_bracket, fold(mutated.primary)[0])
    assert 0 < hamming_distance
    assert 2 == hamming_distance

    environment.design = mutated
    assert 1.0 == environment._get_reward(True)


def test_RnaDesignEnvironment_execute():
    actions = [1, 0, 1, 3, 2, 1, 3]  # Actions correspond to valid solution
    dot_brackets = ["(((....)))"]

    # No conv, no embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=False, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)

    environment.reset()
    states = [[state] for state in environment.target.padded_encoding][
        1:
    ]  # reset() already gets first state

    for index, action in enumerate(actions):
        state, terminal, reward = environment.execute(action)
        if terminal:
            assert None == state
            assert 1.0 == reward
            break
        assert states[index] == state
        assert 0 == reward
        assert False == terminal

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=False, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)

    environment.reset()
    states = [
        [
            environment.target.padded_encoding[i - 1],
            environment.target.padded_encoding[i],
            environment.target.padded_encoding[i + 1],
        ]
        for i in range(len(environment.target.padded_encoding) - 1)
        if i > 0
    ][1:]

    for index, action in enumerate(actions):
        state, terminal, reward = environment.execute(action)
        if terminal:
            assert None == state
            assert 1.0 == reward
            break
        assert states[index] == state
        assert 0 == reward
        assert False == terminal

    # No conv, embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=True, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)

    environment.reset()
    states = [[state] for state in environment.target.padded_encoding][
        1:
    ]  # reset() already gets first state

    for index, action in enumerate(actions):
        state, terminal, reward = environment.execute(action)
        if terminal:
            assert None == state
            assert 1.0 == reward
            break
        assert states[index] == state
        assert 0 == reward
        assert False == terminal

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=True, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)

    environment.reset()
    states = [
        [
            environment.target.padded_encoding[i - 1],
            environment.target.padded_encoding[i],
            environment.target.padded_encoding[i + 1],
        ]
        for i in range(len(environment.target.padded_encoding) - 1)
        if i > 0
    ][1:]

    for index, action in enumerate(actions):
        state, terminal, reward = environment.execute(action)
        if terminal:
            assert None == state
            assert 1.0 == reward
            break
        assert states[index] == state
        assert 0 == reward
        assert False == terminal

    # Conv, no embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=False, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)

    environment.reset()
    states = [[state] for state in environment.target.padded_encoding][
        1:
    ]  # reset() already gets first state

    for index, action in enumerate(actions):
        state, terminal, reward = environment.execute(action)
        if terminal:
            assert None == state
            assert 1.0 == reward
            break
        assert states[index] == state
        assert 0 == reward
        assert False == terminal

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=False, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)

    environment.reset()
    states = [
        [
            environment.target.padded_encoding[i - 1],
            environment.target.padded_encoding[i],
            environment.target.padded_encoding[i + 1],
        ]
        for i in range(len(environment.target.padded_encoding) - 1)
        if i > 0
    ][1:]

    for index, action in enumerate(actions):
        state, terminal, reward = environment.execute(action)
        if terminal:
            assert None == state
            assert 1.0 == reward
            break
        assert states[index] == state
        assert 0 == reward
        assert False == terminal

    # Conv, embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=True, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)

    environment.reset()
    states = [[state] for state in environment.target.padded_encoding][
        1:
    ]  # reset() already gets first state

    for index, action in enumerate(actions):
        state, terminal, reward = environment.execute(action)
        if terminal:
            assert None == state
            assert 1.0 == reward
            break
        assert states[index] == state
        assert 0 == reward
        assert False == terminal

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=True, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)

    environment.reset()
    states = [
        [
            environment.target.padded_encoding[i - 1],
            environment.target.padded_encoding[i],
            environment.target.padded_encoding[i + 1],
        ]
        for i in range(len(environment.target.padded_encoding) - 1)
        if i > 0
    ][1:]

    for index, action in enumerate(actions):
        state, terminal, reward = environment.execute(action)
        if terminal:
            assert None == state
            assert 1.0 == reward
            break
        assert states[index] == state
        assert 0 == reward
        assert False == terminal

    actions = [0, 1, 0, 1, 2, 1]  # Actions correspond to valid solution
    dot_brackets = ["((....))"]

    # No conv, no embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=False, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)

    environment.reset()
    states = [[state] for state in environment.target.padded_encoding][
        1:
    ]  # reset() already gets first state

    for index, action in enumerate(actions):
        state, terminal, reward = environment.execute(action)
        if terminal:
            assert None == state
            assert 1.0 == reward
            break
        assert states[index] == state
        assert 0 == reward
        assert False == terminal

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=False, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)

    environment.reset()
    states = [
        [
            environment.target.padded_encoding[i - 1],
            environment.target.padded_encoding[i],
            environment.target.padded_encoding[i + 1],
        ]
        for i in range(len(environment.target.padded_encoding) - 1)
        if i > 0
    ][1:]

    for index, action in enumerate(actions):
        state, terminal, reward = environment.execute(action)
        if terminal:
            assert None == state
            assert 1.0 == reward
            break
        assert states[index] == state
        assert 0 == reward
        assert False == terminal

    # No conv, embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=True, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)

    environment.reset()
    states = [[state] for state in environment.target.padded_encoding][
        1:
    ]  # reset() already gets first state

    for index, action in enumerate(actions):
        state, terminal, reward = environment.execute(action)
        if terminal:
            assert None == state
            assert 1.0 == reward
            break
        assert states[index] == state
        assert 0 == reward
        assert False == terminal

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=False, use_embedding=True, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)

    environment.reset()
    states = [
        [
            environment.target.padded_encoding[i - 1],
            environment.target.padded_encoding[i],
            environment.target.padded_encoding[i + 1],
        ]
        for i in range(len(environment.target.padded_encoding) - 1)
        if i > 0
    ][1:]

    for index, action in enumerate(actions):
        state, terminal, reward = environment.execute(action)
        if terminal:
            assert None == state
            assert 1.0 == reward
            break
        assert states[index] == state
        assert 0 == reward
        assert False == terminal

    # Conv, no embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=False, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)

    environment.reset()
    states = [[state] for state in environment.target.padded_encoding][
        1:
    ]  # reset() already gets first state

    for index, action in enumerate(actions):
        state, terminal, reward = environment.execute(action)
        if terminal:
            assert None == state
            assert 1.0 == reward
            break
        assert states[index] == state
        assert 0 == reward
        assert False == terminal

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=False, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)

    environment.reset()
    states = [
        [
            environment.target.padded_encoding[i - 1],
            environment.target.padded_encoding[i],
            environment.target.padded_encoding[i + 1],
        ]
        for i in range(len(environment.target.padded_encoding) - 1)
        if i > 0
    ][1:]

    for index, action in enumerate(actions):
        state, terminal, reward = environment.execute(action)
        if terminal:
            assert None == state
            assert 1.0 == reward
            break
        assert states[index] == state
        assert 0 == reward
        assert False == terminal

    # Conv, embedding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=True, state_radius=0
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)

    environment.reset()
    states = [[state] for state in environment.target.padded_encoding][
        1:
    ]  # reset() already gets first state

    for index, action in enumerate(actions):
        state, terminal, reward = environment.execute(action)
        if terminal:
            assert None == state
            assert 1.0 == reward
            break
        assert states[index] == state
        assert 0 == reward
        assert False == terminal

    # Include padding
    environment_config = RnaDesignEnvironmentConfig(
        use_conv=True, use_embedding=True, state_radius=1
    )

    environment = RnaDesignEnvironment(dot_brackets, environment_config)

    environment.reset()
    states = [
        [
            environment.target.padded_encoding[i - 1],
            environment.target.padded_encoding[i],
            environment.target.padded_encoding[i + 1],
        ]
        for i in range(len(environment.target.padded_encoding) - 1)
        if i > 0
    ][1:]

    for index, action in enumerate(actions):
        state, terminal, reward = environment.execute(action)
        if terminal:
            assert None == state
            assert 1.0 == reward
            break
        assert states[index] == state
        assert 0 == reward
        assert False == terminal
