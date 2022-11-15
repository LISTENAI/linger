#!/bin/bash

set -e

pip install -r requirements.txt

pip uninstall linger -y
if [ -e "build" ];then
rm -rf build
fi
MAX_JOBS=32 python setup.py sdist

MAX_JOBS=32 pip install dist/*.gz

pytest test -s