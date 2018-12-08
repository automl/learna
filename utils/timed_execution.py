import pynisher
import subprocess
from pathlib import Path


def execution(method, sequence_path, run_path, target_id):
    run_path.mkdir(parents=True, exist_ok=True)
    time_path = run_path.joinpath(f"{target_id}.time")
    error_path = run_path.joinpath(f"{target_id}.error")
    out_path = run_path.joinpath(f"{target_id}.out")
    try:
        out = subprocess.run(
            f"./utils/execution_scripts/{method}.sh {sequence_path}",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf8",
        )
        timing = out.stderr.split("\n")[-2]  # Assumes timing command in calling method
        time_path.write_text(timing)
        out_path.write_text(out.stdout)
    except subprocess.CalledProcessError as error:
        # No output is captured, why?
        if 9 != error.returncode != 137:  # Not killed by pynisher
            error_path.write_text(error.stderr)
    except:
        error_path.write_text("some other error occured")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeout", type=int, required=True, help="Maximum allowed CPU time"
    )
    parser.add_argument("--data_dir", required=True, help="Root dir of the data")
    parser.add_argument("--results_dir", required=True, help="Root dir for results")
    parser.add_argument("--experiment_group", required=True, help="Group of experiments")
    parser.add_argument("--method", required=True, help="Method name to call")
    parser.add_argument("--dataset", required=True, help="Dataset to use")
    parser.add_argument("--task_id", type=int, required=True, help="Id of the Task")
    args = parser.parse_args()

    dataset_size = len(list(Path(args.data_dir, args.dataset).glob("*.rna")))
    run = (args.task_id - 1) // dataset_size  # 100 / 100 = 1, but should be 0
    target_id = ((args.task_id - 1) % dataset_size) + 1  # X.rna is 1 indexed

    run_path = Path(
        args.results_dir, args.experiment_group, args.dataset, args.method, f"run-{run}"
    )
    sequence_path = Path(args.data_dir, args.dataset, f"{target_id}.rna")

    limited_execution = pynisher.enforce_limits(cpu_time_in_s=args.timeout)(execution)
    limited_execution(args.method, sequence_path, run_path, target_id)
