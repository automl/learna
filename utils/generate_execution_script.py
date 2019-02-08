from pathlib import Path
from utils.configs_to_validation import (
    _validate_config,
    _fill_config,
    _write_config,
    modes,
    timeouts,
)


def generate_execution_script(config_id, config, job_id, output_dir, mode, restore_path):
    if not _validate_config(config):
        config = _fill_config(config, mode)

    with open(
        output_dir.joinpath("execution_scripts/" + job_id + "_" + config_id + ".sh"), "w"
    ) as execution_script:
        execution_script.write("#!/bin/bash\n")
        execution_script.write("TARGET_STRUCTURE_PATH=$1\n")
        execution_script.write("\n")
        execution_script.write(
            "source thirdparty/miniconda/miniconda/bin/activate learna\n"
        )
        execution_script.write(
            '/usr/bin/time -f"%U" python -m src.learna.design_rna \\\n'
        )
        execution_script.write("  --mutation_threshold 5 \\\n")

        _write_config(config, execution_script)

        execution_script.write("  --target_structure_path $TARGET_STRUCTURE_PATH \\\n")
        if mode == "meta_learna" or "meta_learna_adapt":
            execution_script.write(f"  --restore_path {restore_path} ")
            if mode == "meta_learna":
                execution_script.write("\\\n")
                execution_script.write("  --stop_learning")


if __name__ == "__main__":
    import argparse
    import ast

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=Path, help="Directory were the files will be stored"
    )
    parser.add_argument(
        "--job_id", default="12345", help="Id of the job that produced the config"
    )
    parser.add_argument(
        "--mode", default="learna", help="one of learna, meta_learna, meta_learna_adapt"
    )
    parser.add_argument(
        "--root_dir", type=Path, help="Define if validation pipeline file is produced"
    )
    parser.add_argument(
        "--config_id", default="(0, 0, 0)", help="Id of the config received from BOHB"
    )
    parser.add_argument("--restore_path", type=Path, help="The path to load model from")
    parser.add_argument("--config", type=str, help="The config")

    args = parser.parse_args()

    config_id = args.config_id
    config_id = config_id.replace("(", "")
    config_id = config_id.replace(")", "")
    config_id = config_id.replace(",", "_")
    config_id = "".join(config_id.split())

    config = ast.literal_eval(args.config)

    generate_execution_script(
        config_id, config, args.job_id, args.output_dir, args.mode, args.restore_path
    )
