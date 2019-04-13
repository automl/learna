import pandas as pd
from pathlib import Path

sequence_count = {"eterna": 100, "rfam_taneda": 29, "rfam_learn_test": 100}
number_of_evaluations = {"eterna": 5, "rfam_taneda": 50, "rfam_learn_test": 5}


class ExperimentGroupResults(object):
    def __init__(self, data_path, results_path):
        self._group_name = data_path.stem
        self._datasets = {}
        self._read_datasets(data_path)
        self._results_path = results_path

    def _read_datasets(self, data_path):
        for path in data_path.glob("*/"):
            self._datasets[path.stem] = _DatasetResults(path)

    def to_latex(self, column_format=None, compile_ready=False):
        mode = "a" if compile_ready else "w"
        for dataset in self._datasets:
            _quantiles_file = self._results_path.joinpath(
                f"{self._datasets[dataset].name}_quantiles.tex"
            )
            _times_file = self._results_path.joinpath(
                f"{self._datasets[dataset].name}_times.tex"
            )
            _solved_instances_file = self._results_path.joinpath(
                f"{self._datasets[dataset].name}_solved_instances.tex"
            )

            _quantiles_df = self._datasets[dataset].quantiles.fillna(0)
            _times_df = self._datasets[dataset].times.fillna(0)
            _solved_instances_df = self._datasets[dataset].solved_instances.fillna("-")

            if compile_ready:
                [
                    self._write_header(mode, _file)
                    for _file in [_quantiles_file, _times_file, _solved_instances_file]
                ]

            [
                self._write_df_to_table(df, _file, column_format, mode)
                for df, _file in zip(
                    [_quantiles_df, _times_df, _solved_instances_df],
                    [_quantiles_file, _times_file, _solved_instances_file],
                )
            ]

            if compile_ready:
                [
                    self._write_footer(mode, _file)
                    for _file in [_quantiles_file, _times_file, _solved_instances_file]
                ]

    def _write_df_to_table(self, df, tex_file, column_format, mode):
        with open(tex_file, mode) as f:
            f.write(df.to_latex(column_format=column_format))

    def _write_header(self, mode, tex_file):
        with open(tex_file, mode) as f:
            f.write("\\documentclass{article}")
            f.write("\\usepackage{booktabs}")
            f.write("\\begin{document}")

    def _write_footer(self, mode, tex_file):
        with open(tex_file, mode) as f:
            f.write("\\end{document}")

    @property
    def datasets(self):
        return self._datasets


class _DatasetResults(object):
    def __init__(self, data_path):
        self._dataset_name = data_path.stem

        self._methods = {}
        self._read_methods(data_path)

        self._quantiles = pd.DataFrame()
        self._times = pd.DataFrame()
        self._solved_instances = pd.DataFrame()

        self._methods_results_to_df()

    def _read_methods(self, data_path):
        for path in data_path.glob("*/"):
            self._methods[path.stem] = _MethodResults(path)

    def _methods_results_to_df(self):
        for method in self._methods:
            self._quantiles = self._quantiles.append(self._methods[method].run_quantiles)
            self._times = self._times.append(self._methods[method].time_limits)
            self._solved_instances = self._solved_instances.append(
                self._methods[method].solved_instances
            )

    @property
    def name(self):
        return self._dataset_name

    @property
    def methods(self):
        return self._methods

    @property
    def quantiles(self):
        return self._quantiles

    @property
    def times(self):
        return self._times

    @property
    def solved_instances(self):
        df = self._solved_instances.T
        df.loc["Total"] = df.sum()
        return df


class _MethodResults(object):
    def __init__(self, data_path):
        self._method_name = data_path.stem

        self._read_quantiles(data_path)
        self._read_solved_instances(data_path)
        self._read_time_limits(data_path)

    def _read_quantiles(self, data_path):
        _run_quantiles = {}
        with open(data_path.joinpath("run_quantiles.tsv")) as _quantiles:
            lines = _quantiles.readlines()
            for line in lines:
                if line.split()[0] == "quantile":
                    continue
                runs = int(
                    (int(line.split()[0]) / 100)
                    * number_of_evaluations[data_path.parent.stem]
                )
                column_label = "Runs" if runs > 1 else "Run"
                _run_quantiles[f"{runs} {column_label}" if runs != 0 else "Total"] = int(
                    round(
                        (int(line.split()[1]) / sequence_count[data_path.parent.stem])
                        * 100,
                        0,
                    )
                )
        self._run_quantiles = pd.DataFrame(_run_quantiles, index=[self._method_name])

    def _read_solved_instances(self, data_path):
        _solved_instances = {}

        with open(data_path.joinpath("runs_solve_instance.tsv")) as _instances:
            lines = _instances.readlines()
            for line in lines:
                if line.split()[0] == "id":
                    continue
                _solved_instances[int(line.split()[0])] = (
                    str(int(line.split()[1]))
                    + f"/{number_of_evaluations[data_path.parent.stem]}"
                )
        self._solved_instances = pd.DataFrame(
            _solved_instances,
            index=[self._method_name],
            columns=range(1, sequence_count[data_path.parent.stem] + 1),
        )

    def _read_time_limits(self, data_path):
        _time_limits = {}
        with open(data_path.joinpath("time_limit.tsv")) as _times:
            lines = _times.readlines()
            for line in lines:
                if line.split()[0] == "time":
                    continue
                _time_limits[f"{int(line.split()[0])} s"] = int(
                    round(
                        (int(line.split()[1]) / sequence_count[data_path.parent.stem])
                        * 100
                    )
                )
        self._time_limits = pd.DataFrame(_time_limits, index=[self._method_name])

    @property
    def run_quantiles(self):
        return self._run_quantiles.reindex_axis(
            sorted(
                self._run_quantiles.columns,
                key=lambda x: int(x.split()[0]) if x.split()[0] != "Total" else -1,
            ),
            axis=1,
        )

    @property
    def solved_instances(self):
        return self._solved_instances.sort_index(axis=0)

    @property
    def sum_solved_instances(self):
        return self._solved_instances.fillna("-").sum(axis=1)

    @property
    def time_limits(self):
        return self._time_limits.reindex_axis(
            sorted(self._time_limits.columns, key=lambda x: int(x.split()[0])), axis=1
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_group", required=True, type=Path, help="Experiment group to analyse"
    )
    parser.add_argument(
        "--results_dir", type=Path, help="Root folder for analysis results"
    )
    parser.add_argument("--column_format", type=str, help="Custom column formatting")
    parser.add_argument(
        "--compile_ready", action="store_true", help="Output files can be compiled"
    )

    args = parser.parse_args()

    results = ExperimentGroupResults(
        args.experiment_group.resolve(), args.results_dir.resolve()
    )
    results.to_latex(column_format=args.column_format, compile_ready=args.compile_ready)
