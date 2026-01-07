pip uninstall linger -y
rm -rf dist/*.gz
if [ -e "build" ];then
rm -rf build
fi

# pip install -r requirements.txt

echo -n "GCC-verion: "
which gcc
echo -n "CUDA-verion: "
which nvcc
echo -n "CMAKE-verion: "
which cmake
MAX_JOBS=32 python3 setup.py sdist #  bdist_wheel
echo "MAX_JOBS-------------"
MAX_JOBS=32 pip install dist/*.gz --no-build-isolation
