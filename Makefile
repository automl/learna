.DEFAULT_GOAL := show-help
SHELL := /bin/bash
PATH := $(PWD)/thirdparty/miniconda/miniconda/bin:$(PATH)


################################################################################
# Utility
################################################################################

## Format using black
format:
	source activate learna && \
	black src utils -l 90

## Run the testsuite
test:
	@source activate learna && \
	pytest . -p no:warnings

## To clean project state
clean: clean-runtime clean-data clean-thirdparty clean-models clean-results

## Remove runtime files
clean-runtime:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '__pycache__' -exec rm -rf --force {} +

## Remove data files
clean-data:
	rm -rf data/eterna/*.rna
	rm -rf data/eterna/raw/*.txt
	rm -rf data/eterna/interim/*.txt
	rm -rf data/rfam_taneda
	rm -rf data/rfam_learn

## Remove model examples
clean-models:
	rm -rf models/example

## Clean results directory
clean-results: clean-bohb clean-timed-execution
	rm -rf results

## Remove all files from timed execution examples
clean-timed-execution:
	rm -rf results/timed_execution_example

## Remove all bohb-examples
clean-bohb:
	rm -rf results/*.pkl
	rm -rf results/*.json

## Remove thirdparty installs
clean-thirdparty:
	rm -rf thirdparty/miniconda/miniconda

################################################################################
# Setup General
################################################################################

## Download and prepare all datasets
data: data-eterna data-rfam-taneda data-rfam-learn

## Download and make the Eterna100 dataset
data-eterna:
	@source activate learna && \
	python -m src.data.download_and_build_eterna
	./src/data/secondaries_to_single_files.sh data/eterna data/eterna/interim/eterna.txt

## Download and build the Rfam-Taneda dataset
data-rfam-taneda:
	@./src/data/download_and_build_rfam_taneda.sh

## Download and build the Rfam-Learn dataset
data-rfam-learn:
	@./src/data/download_and_build_rfam_learn.sh


################################################################################
# Setup LEARNA
################################################################################

## Install all dependencies
requirements:
	./thirdparty/miniconda/make_miniconda.sh
	conda env create -f environment.yml


################################################################################
# Test Experiment and Example
################################################################################

## Local experiment testing
experiment-test:
	@source activate learna && \
	python -m src.learna.design_rna \
	--mutation_threshold 5 \
  --batch_size 78 \
  --conv_size 0 7 \
  --conv_channels 22 22 \
  --embedding_size 0 \
  --entropy_regularization 0.00010469282668627605 \
  --fc_units 34 \
  --learning_rate 0.00015149356071984718 \
  --lstm_units 36 \
  --num_fc_layers 1 \
  --num_lstm_layers 0 \
  --reward_exponent 4.486673165414606 \
  --state_radius 2 \
	--restart_timeout 1800 \
	--target_structure_path data/eterna/2.rna \
	--timeout 30

## Example call for timed execution
timed-execution-example-%:
	@source activate learna && \
	python utils/timed_execution.py \
		--timeout 30 \
		--data_dir data/ \
		--results_dir results/ \
		--experiment_group timed_execution_example \
		--method LEARNA-30min \
		--dataset eterna \
		--task_id $*


################################################################################
# Reproduce Results of LEARNA
################################################################################

## Reproduce LEARNA-30min on <id> (1-100) of Eterna100
reproduce-LEARNA-Eterna-%:
	@source activate learna && \
	python -m src.learna.design_rna \
		--batch_size 79 \
		--conv_channels 10 3 \
		--embedding_size 0 \
		--entropy_regularization 0.0001628733797899296 \
		--fc_units 32 \
		--learning_rate 0.00033766914645516697 \
		--lstm_units 7 \
		--num_fc_layers 1 \
		--num_lstm_layers 2 \
		--reward_exponent 9.437605850994773 \
		--mutation_threshold 5 \
		--conv_sizes 0 3 \
		--state_radius 2 \
		--target_structure_path data/eterna/$*.rna \
		--timeout 86400

## Reproduce LEARNA-10min on <id> (1-29) of Rfam-Taneda
reproduce-LEARNA-Rfam-Taneda-%:
	@source activate learna && \
	python -m src.learna.design_rna \
		--batch_size 32 \
  	--conv_channels 8 1 \
  	--embedding_size 0 \
  	--entropy_regularization 0.00044440579487984737 \
  	--fc_units 52 \
  	--learning_rate 0.000548959271057026 \
  	--lstm_units 4 \
  	--num_fc_layers 1 \
  	--num_lstm_layers 2 \
  	--reward_exponent 5.724874982958563 \
  	--mutation_threshold 5 \
  	--conv_sizes 5 3 \
  	--state_radius 16 \
  	--target_structure_path data/rfam_taneda/$*.rna \
		--timeout 600

## Reproduce LEARNA-30min on <id> (1-100) of Rfam-Learn-Test
reproduce-LEARNA-Rfam-Learn-Test-%:
	@source activate learna && \
	python -m src.learna.design_rna \
		--batch_size 79 \
		--conv_channels 10 3 \
		--embedding_size 0 \
		--entropy_regularization 0.0001628733797899296 \
		--fc_units 32 \
		--learning_rate 0.00033766914645516697 \
		--lstm_units 7 \
		--num_fc_layers 1 \
		--num_lstm_layers 2 \
		--reward_exponent 9.437605850994773 \
		--mutation_threshold 5 \
		--conv_sizes 0 3 \
		--state_radius 2 \
  	--target_structure_path data/rfam_learn/test/$*.rna \
		--timeout 3600


################################################################################
# Reproduce Results of Meta-LEARNA
################################################################################

## Reproduce Meta-LEARNA on <id> (1-100) of Eterna100
reproduce-Meta-LEARNA-Eterna-%:
	@source activate learna && \
	python -m src.learna.design_rna \
		--batch_size 80 \
  	--conv_channels 32 14 \
  	--embedding_size 1 \
  	--entropy_regularization 0.000198389753598839 \
  	--fc_units 9 \
  	--learning_rate 6.374026866356635e-05 \
  	--lstm_units 53 \
  	--num_fc_layers 1 \
  	--num_lstm_layers 0 \
  	--reward_exponent 9.224721807238447 \
  	--mutation_threshold 5 \
  	--conv_size 5 7 \
  	--state_radius 26 \
  	--target_structure_path data/eterna/$*.rna \
  	--restore_path models/trained_models/54_0_2 \
  	--timeout 86400 \
		--stop_learning

## Reproduce Meta-LEARNA on <id> (1-29) of Rfam-Taneda
reproduce-Meta-LEARNA-Rfam-Taneda-%:
	@source activate learna && \
	python -m src.learna.design_rna \
		--batch_size 80 \
  	--conv_channels 32 14 \
  	--embedding_size 1 \
  	--entropy_regularization 0.000198389753598839 \
  	--fc_units 9 \
  	--learning_rate 6.374026866356635e-05 \
  	--lstm_units 53 \
  	--num_fc_layers 1 \
  	--num_lstm_layers 0 \
  	--reward_exponent 9.224721807238447 \
  	--mutation_threshold 5 \
  	--conv_size 5 7 \
  	--state_radius 26 \
  	--target_structure_path data/rfam_taneda/$*.rna \
  	--restore_path models/trained_models/54_0_2 \
  	--timeout 600 \
		--stop_learning

## Reproduce Meta-LEARNA on <id> (1-100) of Rfam-Learn-Test
reproduce-Meta-LEARNA-Rfam-Learn-Test-%:
	@source activate learna && \
	python -m src.learna.design_rna \
		--batch_size 80 \
  	--conv_channels 32 14 \
  	--embedding_size 1 \
  	--entropy_regularization 0.000198389753598839 \
  	--fc_units 9 \
  	--learning_rate 6.374026866356635e-05 \
  	--lstm_units 53 \
  	--num_fc_layers 1 \
  	--num_lstm_layers 0 \
  	--reward_exponent 9.224721807238447 \
  	--mutation_threshold 5 \
  	--conv_size 5 7 \
  	--state_radius 26 \
  	--target_structure_path data/rfam_learn/test/$*.rna \
  	--restore_path models/trained_models/54_0_2 \
  	--timeout 3600 \
		--stop_learning


################################################################################
# Reproduce Results of Meta-LEARNA-Adapt
################################################################################

## Reproduce Meta-LEARNA-Adapt on <id> (1-100) of Eterna100
reproduce-Meta-LEARNA-Adapt-Eterna-%:
	@source activate learna && \
	python -m src.learna.design_rna \
		--batch_size 80 \
  	--conv_channels 32 14 \
  	--embedding_size 1 \
  	--entropy_regularization 0.000198389753598839 \
  	--fc_units 9 \
  	--learning_rate 6.374026866356635e-05 \
  	--lstm_units 53 \
  	--num_fc_layers 1 \
  	--num_lstm_layers 0 \
  	--reward_exponent 9.224721807238447 \
  	--mutation_threshold 5 \
  	--conv_size 5 7 \
  	--state_radius 26 \
  	--target_structure_path data/eterna/$*.rna \
  	--restore_path models/trained_models/54_0_2 \
  	--timeout 86400

## Reproduce Meta-LEARNA-Adapt on <id> (1-29) of Rfam-Taneda
reproduce-Meta-LEARNA-Adapt-Rfam-Taneda-%:
	@source activate learna && \
	python -m src.learna.design_rna \
		--batch_size 80 \
  	--conv_channels 32 14 \
  	--embedding_size 1 \
  	--entropy_regularization 0.000198389753598839 \
  	--fc_units 9 \
  	--learning_rate 6.374026866356635e-05 \
  	--lstm_units 53 \
  	--num_fc_layers 1 \
  	--num_lstm_layers 0 \
  	--reward_exponent 9.224721807238447 \
  	--mutation_threshold 5 \
  	--conv_size 5 7 \
  	--state_radius 26 \
  	--target_structure_path data/rfam_taneda/$*.rna \
  	--restore_path models/trained_models/54_0_2 \
  	--timeout 600

## Reproduce Meta-LEARNA-Adapt on <id> (1-100) of Rfam-Learn-Test
reproduce-Meta-LEARNA-Adapt-Rfam-Learn-Test-%:
	@source activate learna && \
	python -m src.learna.design_rna \
		--batch_size 80 \
  	--conv_channels 32 14 \
  	--embedding_size 1 \
  	--entropy_regularization 0.000198389753598839 \
  	--fc_units 9 \
  	--learning_rate 6.374026866356635e-05 \
  	--lstm_units 53 \
  	--num_fc_layers 1 \
  	--num_lstm_layers 0 \
  	--reward_exponent 9.224721807238447 \
  	--mutation_threshold 5 \
  	--conv_size 5 7 \
  	--state_radius 26 \
  	--target_structure_path data/rfam_learn/test/$*.rna \
  	--restore_path models/trained_models/54_0_2 \
  	--timeout 3600


################################################################################
# Joint Architecture and Hyperparameter Search
################################################################################

## Run an example for joint Hyperparameter and Architecture Search using BOHB
bohb-example:
	@source activate learna && \
	python -m src.optimization.bohb \
	  --min_budget 2 \
		--max_budget 8 \
		--n_iter 1 \
		--n_cores 1 \
		--run_id example \
		--data_dir data \
		--nic_name lo \
		--shared_directory results/ \
		--mode learna

################################################################################
# Run experiments on cluster
################################################################################

## Submitt a test on MOAB cluster%
nemo-test-%:
	msub utils/rna_single.moab \
		-l walltime=100 \
		-t 1-2 \
		-v METHOD=$* \
		-v DATASET=rfam_learn/test \
		-v TIMEOUT=60 \
		-v EXPERIMENT_GROUP=test_rerun


## Start experiment on Rfam-Learn benchmark
nemo-rfam-learn-test-%:
	msub utils/rna_single.moab \
		-l walltime=5000 \
		-t 1-500 \
		-v METHOD=$* \
		-v DATASET=rfam_learn/test \
		-v TIMEOUT=3600 \
		-v EXPERIMENT_GROUP=first_bohb_rerun


## Start experiment on the Rfam Taneda benchmark
nemo-rfam-taneda-%:
	msub utils/rna_single.moab \
		-l walltime=1000 \
		-t 1-1450 \
		-v METHOD=$* \
		-v DATASET=rfam_taneda \
		-v TIMEOUT=600 \
		-v EXPERIMENT_GROUP=first_bohb_rerun


## Start experiment on the Eterna100 benchmark
nemo-eterna-%:
	msub utils/rna_single.moab \
		-l walltime=100000 \
		-t 1-500 \
		-v METHOD=$* \
		-v DATASET=eterna \
		-v TIMEOUT=86400 \
		-v EXPERIMENT_GROUP=first_bohb_rerun

## Generate files for validation pipeline on cluster
validation-files:
	@source activate learna && \
	python -m utils.configs_to_validation \
	  --config "{'batch_size': 80, 'conv_channels1': 32, 'conv_channels2': 14, 'embedding_size': 1, 'entropy_regularization': 0.000198389753598839, 'fc_units': 9, 'learning_rate': 6.374026866356635e-05, 'lstm_units': 53, 'num_fc_layers': 1, 'num_lstm_layers': 0, 'reward_exponent': 9.224721807238447, 'conv_size1': 5, 'conv_size2': 7, 'state_radius': 26}" \
		--config_id "(54, 0, 2)" \
		--job_id 00000 \
		--mode learna \
		--root_dir $(PWD) \
		--num_repeats 2 \
		--num_validations 1


################################################################################
# Help
################################################################################

# From https://drivendata.github.io/cookiecutter-data-science/
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=22 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
