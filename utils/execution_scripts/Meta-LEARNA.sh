#!/bin/bash
TARGET_STRUCTURE_PATH=$1

source thirdparty/miniconda/miniconda/bin/activate learna
/usr/bin/time -f"%U" python -m src.learna.design_rna \
  --mutation_threshold 5 \
  --batch_size 123 \
  --conv_sizes 11 3 \
  --conv_channels 10 3 \
  --embedding_size 2 \
  --entropy_regularization 0.00015087352506343337 \
  --fc_units 52 \
  --learning_rate 6.442010833400271e-05 \
  --lstm_units 3 \
  --num_fc_layers 1 \
  --num_lstm_layers 0 \
  --reward_exponent 8.932893783628236 \
  --state_radius 29 \
  --target_structure_path $TARGET_STRUCTURE_PATH \
  --restore_path models/ICLR_2019/224_0_1 \
  --stop_learning
