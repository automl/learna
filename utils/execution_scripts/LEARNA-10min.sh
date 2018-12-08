#!/bin/bash
TARGET_STRUCTURE_PATH=$1

source thirdparty/miniconda/miniconda/bin/activate learna
/usr/bin/time -f"%U" python -m src.learna.design_rna \
  --batch_size 32 \
  --conv_channels 8 1 \
  --embedding_size 0 \
  --entropy_regularization 0.00044440579487984737 \
  --fc_units 52 \
  --learning_rate 0.000548959271057026 \
  --lstm_units 4 \
  --num_fc_layers 1 \
  --num_lstm_layers 2 \
  --optimization_steps 10 \
  --reward_exponent 5.724874982958563 \
  --mutation_threshold 5 \
  --conv_sizes 5 3 \
  --state_radius 16 \
  --likelihood_ratio_clipping 0.3 \
  --include_mutation \
  --fc_activation relu \
  --target_structure_path $TARGET_STRUCTURE_PATH \
  --restart_timeout 600
