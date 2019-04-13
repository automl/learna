#!/bin/bash
TARGET_STRUCTURE_PATH=$1

source thirdparty/miniconda/miniconda/bin/activate learna
/usr/bin/time -f"%U" python -m src.learna.design_rna \
  --mutation_threshold 5 \
  --batch_size 126 \
  --conv_sizes 17 5 \
  --conv_channels 7 18 \
  --embedding_size 3 \
  --entropy_regularization 6.762991409135427e-05 \
  --fc_units 57 \
  --learning_rate 0.0005991629320464973 \
  --lstm_units 28 \
  --num_fc_layers 1 \
  --num_lstm_layers 1 \
  --reward_exponent 9.33503385734547 \
  --state_radius 32 \
  --target_structure_path $TARGET_STRUCTURE_PATH \
  --restart_timeout 1800
