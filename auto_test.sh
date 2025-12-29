#!/bin/bash

set -e

source ~/anaconda3/bin/activate
conda activate linger_env
module load gcc/5.4-os7
module load cuda/10.2-cudnn-7.6.5
module load cmake/3.17.3
pip install torchvision==0.10.0
pip install pytest
pip install onnx==1.7.0
pip install ninja
pip install protobuf==3.8.0

pip uninstall linger -y
if [ -e "build" ];then
rm -rf build
fi
MAX_JOBS=32 python setup.py sdist

MAX_JOBS=32 pip install dist/*.gz

pytest test -s