#!/bin/bash
TARGET_STRUCTURE_PATH=$1

source thirdparty/miniconda/miniconda/bin/activate learna
/usr/bin/time -f"%U" python -m src.learna.design_rna \
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
  --target_structure_path $TARGET_STRUCTURE_PATH \
  --restore_path models/trained_models/checkpoint_54_0_2 \
  --stop_learning
