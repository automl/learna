from pathlib import Path
import pandas as pd
import numpy as np
import scikits.bootstrap as sci

from .read_data import read_data_from_method_path, read_sequence_lengths
from .process_data import (
    solved_across_time_per_run,
    solved_across_time_min,
    runs_solve_instance,
    solved_per_time_limit,
    solved_per_run_quantile,
    time_across_length,
)


_datasets = {"eterna", "rfam_taneda", "rfam_learn_validation", "rfam_learn_test"}
_dataset_sizes = {
    "eterna": 100,
    "rfam_taneda": 29,
    "rfam_learn_validation": 100,
    "rfam_learn_test": 100,
}
_runs_per_dataset = {
    "eterna": 5,
    "rfam_taneda": 50,
    "rfam_learn_validation": 5,
    "rfam_learn_test": 5,
}
_timeout_per_dataset = {
    "eterna": 86400,
    "rfam_taneda": 600,
    "rfam_learn_validation": 600,
    "rfam_learn_test": 3600,
}
_time_limits_per_dataset = {
    "eterna": (10, 60, 1800, 3600, 14400, 43200, 86400),
    "rfam_taneda": (10, 30, 60, 300, 600),
    "rfam_learn_validation": (10, 30, 60, 300, 600, 1800),
    "rfam_learn_test": (10, 30, 60, 300, 600, 1200, 1800, 3600),
}
_run_quantiles_per_dataset = {
    "eterna": (0, 20, 40, 60, 80, 100),
    "rfam_taneda": (0, 10, 20, 50, 100),
    "rfam_learn_validation": (0, 20, 40, 60, 80, 100),
    "rfam_learn_test": (0, 20, 40, 60, 80, 100),
}


def pd_to_tsv(output_dir, filename, pd, index_label=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_dir.joinpath(filename).open("w") as output_file:
        pd.to_csv(output_file, index_label=index_label, sep="\t")


def analyse_method(method_path, dataset_name, output_dir, sequences_dir, ci_alpha):
    runs, ids, times = read_data_from_method_path(
        method_path, _timeout_per_dataset[dataset_name]
    )

    sequence_lengths = read_sequence_lengths(sequences_dir)
    length_analysis = time_across_length(runs, ids, times, sequence_lengths)
    length_analysis["method"] = method_path.name
    pd_to_tsv(
        output_dir.joinpath(f"{dataset_name}/{method_path.name}/"),
        "length_to_time.tsv",
        length_analysis,
        index_label="length",
    )
    try:
        per_run_analysis = solved_across_time_per_run(
            runs,
            times,
            ci_alpha,
            _dataset_sizes[dataset_name],
            _timeout_per_dataset[dataset_name],
        )
        pd_to_tsv(
            output_dir.joinpath(f"{dataset_name}/{method_path.name}/"),
            f"ci.tsv",
            per_run_analysis,
            index_label="time",
        )
    except:
        pd_to_tsv(
            output_dir.joinpath(f"{dataset_name}/{method_path.name}/"),
            f"ci.tsv",
            pd.DataFrame({'time' : 1e-10, 'high_ci_0.05': 0.0, 'low_ci_0.05' : 0.0, 'mean' : 0.0}, index = [1e-10]),
            index_label="time",
        )

    min_analysis = solved_across_time_min(
        runs, ids, times, _dataset_sizes[dataset_name], _timeout_per_dataset[dataset_name]
    )
    pd_to_tsv(
        output_dir.joinpath(f"{dataset_name}/{method_path.name}/"),
        f"min.tsv",
        min_analysis,
        index_label="time",
    )

    # number of runs solving individual instances
    runs_solve_instance_analysis = runs_solve_instance(runs, ids)
    pd_to_tsv(
        output_dir.joinpath(f"{dataset_name}/{method_path.name}/"),
        f"runs_solve_instance.tsv",
        runs_solve_instance_analysis,
        index_label="id",
    )

    # number of structures solved within a given time limit by at least one run
    # TODO(Frederic): get time_limits from command line
    time_limit_analysis = solved_per_time_limit(
        runs, ids, times, time_limits=_time_limits_per_dataset[dataset_name]
    )
    pd_to_tsv(
        output_dir.joinpath(f"{dataset_name}/{method_path.name}/"),
        f"time_limit.tsv",
        time_limit_analysis,
        index_label="time",
    )

    # Number of structures solved in X% of the runs
    # TODO(frederic): get quantiles via command line
    solved_per_runs_quantile = solved_per_run_quantile(
        runs,
        ids,
        _runs_per_dataset[dataset_name],
        quantiles=_run_quantiles_per_dataset[dataset_name],
    )
    pd_to_tsv(
        output_dir.joinpath(f"{dataset_name}/{method_path.name}/"),
        f"run_quantiles.tsv",
        solved_per_runs_quantile,
        index_label="quantile",
    )


def analyse_dataset(dataset_path, output_dir, root_sequences_dir, ci_alpha):
    for method_path in dataset_path.iterdir():
        analyse_method(
            method_path,
            dataset_path.name,
            output_dir,
            root_sequences_dir.joinpath(dataset_path.name),
            ci_alpha,
        )


def analyse_experiment_group(
    experiment_group, analysis_dir, root_sequences_dir, ci_alpha
):
    for path in experiment_group.iterdir():
        if path.name in _datasets:
            analyse_dataset(
                path, analysis_dir or experiment_group, root_sequences_dir, ci_alpha
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_group", required=True, type=Path, help="Experiment group to analyse"
    )
    parser.add_argument(
        "--analysis_dir", type=Path, help="Root folder for analysis results"
    )
    parser.add_argument(
        "--root_sequences_dir", type=Path, help="Root folder for datasets"
    )
    parser.add_argument(
        "--ci_alpha", default=0.05, type=float, help="Alpha for confidence intervalls"
    )
    args = parser.parse_args()

    analyse_experiment_group(
        args.experiment_group, args.analysis_dir, args.root_sequences_dir, args.ci_alpha
    )
