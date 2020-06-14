from pathlib import Path
from collections import defaultdict

class Results(object):
    def __init__(self, experiment_group, results_dir):
        self._experiment_group = experiment_group.resolve()
        self._results_dir = results_dir
        self._datasets = [dataset.stem for dataset in self._experiment_group.glob('*/')]
        self._dataset_methods = {}
        for dataset in self._datasets:
            self._dataset_methods[dataset] = self._experiment_group.glob(f"{dataset}/*/")

    def _read_data(self, dataset, file_type):
        return [path.resolve() for path in self._experiment_group.glob(f"**/{file_type}.tsv") if path.parent.parent.stem == dataset]

    def to_latex(self):
        self._write_preamble()
        for dataset in self._datasets:
            self._add_plots(dataset)

    def _add_plots(self, dataset):
        min_data = self._read_data(dataset, 'min')
        ci_data = self._read_data(dataset, 'ci')
        length_plot_data = self._read_data(dataset, 'length_to_time')

        self._write_performance_data(dataset, min_data, ci_data)
        # self._write_length_plot_data(length_plot_data)
        with open(self._results_dir / 'plots.tex', 'a') as plotting_file:
            plotting_file.write(
                "\\end{document}"
            )

    def _write_performance_data(self, dataset, min_data, ci_data):
        subsection = "\\subsection" + "{" + dataset[0].upper() + dataset[1:] + "}\n"
        with open(self._results_dir / 'plots.tex', 'a') as plotting_file:
            plotting_file.write("\\section{Performance}\n" + subsection)

        self._write_tikz_env_start(dataset)
        exists_ci_data = bool([method.stem for method in self._dataset_methods[dataset] if method.stem in [path.parent.stem for path in ci_data]])

        for path in min_data:
            method = path.parent.stem
            with open(self._results_dir / 'plots.tex', 'a') as plotting_file:
                style = f"{''.join(method.split('-')).lower() if method != 'Meta-LEARNA-Adapt' else 'metalearna adapt'}"
                plotting_file.write(
                "\\addplot [" + style + " style] table {" + str(path) + "};\n"
                )
                # check if ci data available, if not, write legend entry
                if not exists_ci_data:
                    legend_entry = f"{path.parent.stem}"
                    plotting_file.write(
                        "\\addlegendentry{" + legend_entry + "}\n"
                        "\n"
                    )

        self._write_tikz_env_end(dataset)

        if exists_ci_data:
            self._write_tikz_env_start(dataset)

            for path in ci_data:
                method = path.parent.stem
                with open(self._results_dir / 'plots.tex', 'a') as plotting_file:
                    style = f"{''.join(method.split('-')).lower() if method != 'Meta-LEARNA-Adapt' else 'metalearna adapt'}"
                    legend_entry = f"{path.parent.stem}"
                    plotting_file.write(
                        "\\plotwithconfidence{" + str(path) + "}{time}{mean}{low_ci_0.05}{high_ci_0.05}{" + style + " style}\n"
                        "\\addlegendentry{" + legend_entry + "}\n"
                        "\n"
                    )

            self._write_tikz_env_end(dataset)
        with open(self._results_dir / 'plots.tex', 'a') as plotting_file:
            plotting_file.write(
                "\n"
            )

    def _write_tikz_env_start(self, dataset):
        with open(self._results_dir / 'plots.tex', 'a') as plotting_file:
            plotting_file.write(
                "% ========================================================================================\n"
                "\\begin{tikzpicture}\n"
                "\\begin{semilogxaxis}[\n"
                "    xlabel={Time [seconds]},\n"
                "    ylabel={Solved Sequences [\\%]},\n"
                "    ymin=0, ymax=100,\n"
                "    xmin=1, xmax=100000,\n"
                "    legend cell align=left,\n"
                "    legend pos= outer north east,\n"
                "    axis line style = thick\n"
                "]\n"
                )


    def _write_tikz_env_end(self, dataset):
        with open(self._results_dir / 'plots.tex', 'a') as plotting_file:
            plotting_file.write(
                "\\end{semilogxaxis}\n"
                "\\end{tikzpicture}\n"
                "%\n"
                "%\n"
                "%\n"
                "%"
            )

    def _write_preamble(self):
        with open(self._results_dir / 'plots.tex', 'w+') as plotting_file:
            plotting_file.write(
            "\\documentclass{article}\n"
            "\n"
            "\\usepackage{visualization/packages/utils}\n"
            "\\usepackage{visualization/packages/iclr2019_conference}\n"
            "\\usepackage{visualization/packages/math_commands}\n"
            "\n"
            "% ________________________________________________________________________________________\n"
            "% Head/Footmatter\n"
            "% ________________________________________________________________________________________\n"
            "\\title{Learning to Design RNA -- Reproduce}\n"
            "\n"
            "\\author{Frederic Runge%\\thanks{Frederic Runge and Danny Stoll contributed equally to this work; order determined by coinflip.}\n"
            ", Danny Stoll%\\footnotemark[1]\n"
            ", Stefan Falkner\n"
            "\& Frank Hutter \\\\\n"
            "Department of Computer Science \\\\\n"
            "University of Freiburg \\\\\n"
            "\\texttt{\\{runget,stolld,sfalkner,fh\}@cs.uni-freiburg.de}}\n"
            "\n"
            "\\newcommand{\\fix}{\marginpar{FIX}}\n"
            "\\newcommand{\\new}{\marginpar{NEW}}\n"
            "\n"
            "\\iclrfinalcopy % Uncomment for camera-ready version, but NOT for submission.\n"
            "\n"
            "\\usepackage{pgfplots}\n"
            "\\pgfplotsset{compat=1.13}\n"
            "\n"
            "\\usepgfplotslibrary{fillbetween}\n"
            "\\usepgfplotslibrary{external}\n"
            "\\usepgfplotslibrary{colorbrewer}\n"
            "\n"
            "\\pgfplotsset{\n"
            "tick label style={font=\\footnotesize},\n"
            "label style={font=\\small},\n"
            "legend style={font=\\tiny},\n"
            "height=5cm,\n"
            "width=0.415\\linewidth,\n"
            "filter discard warning=false,\n"
            "clip marker paths=true,\n"
            "grid=both,\n"
            "title style={at={(0.5,0.95)}, anchor=south},\n"
            "}\n"
            "\n"
            "\\pgfplotsset{learna style/.style={            color=black,  line width=0.75, mark size=1.5, mark=none, const plot}}\n"
            "\\pgfplotsset{metalearna style/.style={        color=Set1-B, line width=0.75, mark size=1.5, mark=none, const plot}}\n"
            "\\pgfplotsset{metalearna adapt style/.style={  color=Set1-E, line width=0.75, mark size=1.5, mark=none, const plot}}\n"
            "\\pgfplotsset{mcts style/.style={              color=Set1-D, line width=0.75, mark size=1.5, mark=none, const plot}}\n"
            "\\pgfplotsset{eastman style/.style={           color=Set1-C, line width=0.75, mark size=1.5, mark=none, const plot}}\n"
            "\\pgfplotsset{antarna style/.style={           color=Set1-A, line width=0.75, mark size=1.0, mark=none, const plot}}\n"
            "\\pgfplotsset{rnainverse style/.style={        color=Set1-G, line width=0.75, mark size=1.5, mark=none, const plot}}\n"
            "\n"
            "% style for the confidence interval\n"
            "\\pgfplotsset{ci style/.style={fill opacity=.5,line width=0, draw=none, mark=none}}\n"
            "\n"
            "\\pgfplotsset{grid style={help lines, color=white!80!black}}\n"
            "\n"
            "\\pgfplotsset{\n"
            "legend image code/.code={\n"
            "\\draw[mark repeat=2,mark phase=2]\n"
            "plot coordinates {\n"
            "(0cm,0cm)\n"
            "(0.15cm,0cm)        %% default is (0.3cm,0cm)\n"
            "(0.3cm,0cm)         %% default is (0.6cm,0cm)\n"
            "};%\n"
            "}\n"
            "}\n"
            "\n"
            "\\newcommand{\\plotwithconfidence}[6]{%\n"
            "    \\pgfplotstableread{#1}\datatable\n"
            "    \\addplot [const plot, name path=upper, ci style,forget plot]\n"
            "         table [x={#2},y={#5}] {\datatable};\n"
            "\n"
            "\\addplot [const plot, name path=lower,ci style,forget plot]\n"
            "    table [x={#2},y={#4}] {\datatable};\n"
            "\n"
            "\\addplot [const plot, fill,  #6, ci style, forget plot]\n"
            "    fill between[of=upper and lower];\n"
            "\n"
            "\\addplot [#6] table [x={#2}, y={#3}]{\datatable};\n"
            "}\n"
            "\n"
            "\\begin{document}\n"
            "\\maketitle\n"
            )



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_group", required=True, type=Path, help="Experiment group to analyse"
    )
    parser.add_argument(
        "--results_dir", required=True, type=Path, help="Directory to save results to"
    )

    args = parser.parse_args()

    results = Results(args.experiment_group, args.results_dir)
    results.to_latex()
