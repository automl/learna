from pathlib import Path


modes = ["learna", "meta_learna", "meta_learna_adapt"]
timeouts = {"learna": 600, "meta_learna_adapt": 1800}


def _write_config(config, execution_script):
    conv_written = False
    for entry in config:
        if "conv" in entry and not conv_written:
            conv_written = True
            execution_script.write(
                f"  --conv_sizes {config['conv_size1']} {config['conv_size2']} \\\n"
            )
            execution_script.write(
                f"  --conv_channels {config['conv_channels1']} {config['conv_channels2']} \\\n"
            )
            continue
        elif "conv" in entry and conv_written:
            continue
        else:
            execution_script.write(f"  --{entry} {config[entry]} \\\n")


def generate_validation_pipeline_file(config_id, config, job_id, mode, root_dir):
    if not _validate_config(config):
        config = _fill_config(config, mode)

    with open(
        root_dir.joinpath("utils/validation_" + job_id + "_" + config_id + ".moab"), "w"
    ) as validation_script:
        validation_script.write("#!/bin/bash\n")
        validation_script.write("#MSUB -N rna_validation\n")
        validation_script.write("#MSUB -E\n")
        validation_script.write("#MSUB -e logs/${MOAB_JOBID}.e\n")
        validation_script.write("#MSUB -o logs/${MOAB_JOBID}.o\n")
        validation_script.write("#MSUB -l nodes=1:ppn=20, walltime=0:00:15:00\n")
        validation_script.write("#MSUB -l pmem=5gb\n")
        validation_script.write("#MSUB -t 1-2\n")
        validation_script.write("\n")

        validation_script.write("mkdir $TMPDIR/${MOAB_JOBID}\n")
        validation_script.write("mkdir $TMPDIR/${MOAB_JOBID}/models\n")
        validation_script.write(
            "tar -xzvf /work/ws/nemo/fr_ds371-learna-0/data/rfam_learn.tar.gz -C $TMPDIR/${MOAB_JOBID}\n"
        )
        validation_script.write("\n")

        validation_script.write(f"cd {root_dir}\n")
        validation_script.write("WORKSPACE=$(cat utils/workspace.txt)\n")
        validation_script.write('DATA_DIR="$TMPDIR/${MOAB_JOBID}"\n')
        validation_script.write('RESULTS_DIR="${WORKSPACE}/results"\n')
        validation_script.write("\n")

        validation_script.write(
            f"source {root_dir}/thirdparty/miniconda/miniconda/bin/activate learna\n"
        )
        validation_script.write("python -m src.learna.learn_to_design_rna \\\n")
        validation_script.write("  --data_dir ${DATA_DIR} \\\n")
        validation_script.write("  --dataset rfam_learn/train \\\n")
        validation_script.write(
            "  --save_path $TMPDIR/${MOAB_JOBID}/models/${MOAB_JOBARRAYINDEX}/ \\\n"
        )
        validation_script.write("  --timeout 30 \\\n")
        validation_script.write("  --worker_count 20 \\\n")
        validation_script.write("  --mutation_threshold 5 \\\n")

        _write_config(config, validation_script)

        output_dir = root_dir.joinpath("utils/")

        validation_script.write("\n")
        validation_script.write(
            f"python -m utils.generate_execution_script --config_id {config_id} "
            + '--config "'
            + f"{config}"
            + '" '
            + "--job_id validation_${MOAB_JOBARRAYINDEX}_"
            + f"{job_id} --mode meta_learna --output_dir {output_dir} "
            + "--restore_path $TMPDIR/${MOAB_JOBID}/models/${MOAB_JOBARRAYINDEX}/"
            + "\n"
        )
        validation_script.write(
            f"chmod a+rwx {root_dir}"
            + "/utils/execution_scripts/validation_${MOAB_JOBARRAYINDEX}_"
            + f"{job_id}_{config_id}.sh"
            + "\n"
        )
        validation_script.write("\n")

        validation_script.write("i=1; while [[ i -le 1 ]];\n")
        validation_script.write("do \\\n")
        validation_script.write("$(python utils/timed_execution.py \\\n")
        validation_script.write("  --timeout 30 \\\n")
        validation_script.write("  --data_dir $TMPDIR/${MOAB_JOBID} \\\n")
        validation_script.write("  --results_dir $RESULTS_DIR \\\n")
        validation_script.write("  --experiment_group validation \\\n")
        validation_script.write(
            "  --method validation_${MOAB_JOBARRAYINDEX}_"
            + f"{job_id}_{config_id}"
            + "\\\n"
        )
        validation_script.write("  --dataset rfam_learn/validation \\\n")
        validation_script.write("  --task_id $i);\n")
        validation_script.write("let i=$i+1; done\n")

        validation_script.write("\n")
        validation_script.write(
            "cp -r $TMPDIR/${MOAB_JOBID}/models/${MOAB_JOBARRAYINDEX}/"
            + f" {root_dir}/models/"
            + "validation_${MOAB_JOBARRAYINDEX}_"
            + f"{job_id}_{config_id}"
        )


def _validate_config(config):
    return not "state_radius_relative" in config


def _fill_config(config, mode):
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

    if mode in timeouts:
        config["restart_timeout"] = timeouts[mode]

    return config


if __name__ == "__main__":
    import argparse
    import ast

    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--config", default=None, help="The configuration received from BOHB"
    )

    args = parser.parse_args()

    config_id = args.config_id
    config_id = config_id.replace("(", "")
    config_id = config_id.replace(")", "")
    config_id = config_id.replace(",", "_")
    config_id = "".join(config_id.split())

    config = ast.literal_eval(args.config)

    generate_validation_pipeline_file(
        config_id, config, args.job_id, args.mode, args.root_dir.resolve()
    )
