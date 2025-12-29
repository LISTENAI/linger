pip uninstall linger -y
rm -rf dist/*.gz
if [ -e "build" ];then
rm -rf build
fi

# pip install -r requirements.txt

module unload gcc
module unload cuda
module unload cmake

module load gcc/10.2.0
module load cuda/11.8-cudnn-v8.9.0

# module load gcc/5.4-os7
# module load cuda/10.2-cudnn-7.6.5

module load cmake/3.29.0
echo -n "GCC-verion: "
which gcc
echo -n "CUDA-verion: "
which nvcc
echo -n "CMAKE-verion: "
which cmake
MAX_JOBS=32 python3 setup.py sdist #  bdist_wheel
echo "MAX_JOBS-------------"
MAX_JOBS=32 pip install dist/*.gz --no-build-isolation
