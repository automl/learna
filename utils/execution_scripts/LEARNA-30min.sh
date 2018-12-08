#!/bin/bash
TARGET_STRUCTURE_PATH=$1

source thirdparty/miniconda/miniconda/bin/activate learna
/usr/bin/time -f"%U" python -m src.learna.design_rna \
  --batch_size 79 \
  --conv_channels 10 3 \
  --embedding_size 0 \
  --entropy_regularization 0.0001628733797899296 \
  --fc_units 32 \
  --learning_rate 0.00033766914645516697 \
  --lstm_units 7 \
  --num_fc_layers 1 \
  --num_lstm_layers 2 \
  --optimization_steps 10 \
  --reward_exponent 9.437605850994773 \
  --mutation_threshold 5 \
  --include_mutation \
  --conv_sizes 0 3 \
  --restart_timeout 1800 \
  --state_radius 2 \
  --likelihood_ratio_clipping 0.3 \
  --fc_activation relu \
  --target_structure_path $TARGET_STRUCTURE_PATH
