#!/bin/bash

cd thirdparty/miniconda

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh
bash install_miniconda.sh -b -p ./miniconda  # Silent mode
rm install_miniconda.sh
