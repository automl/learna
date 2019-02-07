from pathlib import Path

modes = ['learna', 'meta_learna', 'meta_learna_adapt']
timeouts = {'learna': 600, 'meta_learna_adapt': 1800}


def generate_execution_script(config_id, config, job_id, output_dir, mode, restore_path):
    conv_written = False
    if not _validate_config(config):
        config = _fill_config(config, mode)

    with open(output_dir.joinpath('execution_scripts/' + job_id + '_' + config_id + '.sh'), 'w') as execution_script:
        execution_script.write('#!/bin/bash\n')
        execution_script.write('TARGET_STRUCTURE_PATH=$1\n')
        execution_script.write('\n')
        execution_script.write('source thirdparty/miniconda/miniconda/bin/activate learna\n')
        execution_script.write('/usr/bin/time -f"%U" python -m src.learna.design_rna \\\n')
        execution_script.write('  --mutation_threshold 5 \\\n')

        for entry in config:
            if 'conv' in entry and not conv_written:
                conv_written = True
                execution_script.write(f"  --conv_sizes {config['conv_size1']} {config['conv_size2']} \\\n")
                execution_script.write(f"  --conv_channels {config['conv_channels1']} {config['conv_channels2']} \\\n")
                continue
            elif 'conv' in entry and conv_written:
                continue
            else:
                execution_script.write(f"  --{entry} {config[entry]} \\\n")

        execution_script.write('  --target_structure_path $TARGET_STRUCTURE_PATH \\\n')
        if mode == 'meta_learna' or 'meta_learna_adapt':
            execution_script.write(f"  --restore_path {restore_path} ")
            if mode == 'meta_learna':
                execution_script.write('\\\n')
                execution_script.write('  --stop_learning')


def _validate_config(config):
    return not 'state_radius_relative' in config


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
        del config['state_radius_relative']
    else:
        min_state_radius = config["conv_size2"] + config["conv_size2"] - 1
        max_state_radius = 32
        config["state_radius"] = int(
            min_state_radius
            + (max_state_radius - min_state_radius) * config["state_radius_relative"]
        )
        del config['state_radius_relative']


    if mode in timeouts:
        config["restart_timeout"] = timeouts[mode]

    return config


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=Path, help="Directory were the files will be stored"
    )
    parser.add_argument("--job_id", default='12345', help="Id of the job that produced the config")
    parser.add_argument("--mode", default='learna', help="one of learna, meta_learna, meta_learna_adapt")
    parser.add_argument("--root_dir", type=Path, help="Define if validation pipeline file is produced")
    parser.add_argument("--config_id", default='(0, 0, 0)', help="Id of the config received from BOHB")
    parser.add_argument("--restore_path", type=Path, help="The path to load model from")
    parser.add_argument("--config", type=str, help="The config")

    args = parser.parse_args()

    config_id = args.config_id
    config_id = config_id.replace('(', '')
    config_id = config_id.replace(')', '')
    config_id = config_id.replace(',', '_')
    config_id = ''.join(config_id.split())

    config = {'batch_size': 78, 'conv_channels1': 22, 'conv_channels2': 22, 'conv_radius1': 0, 'conv_radius2': 3, 'embedding_size': 0, 'entropy_regularization': 0.00010469282668627605, 'fc_units': 34, 'learning_rate': 0.00015149356071984718, 'lstm_units': 36, 'num_fc_layers': 1, 'num_lstm_layers': 0, 'reward_exponent': 4.486673165414606, 'state_radius_relative': 0.11994062473492281}
    print(config_id)

    generate_execution_script(config_id, config, args.job_id, args.output_dir, args.mode, args.restore_path)
