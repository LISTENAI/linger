pip uninstall linger -y
rm -rf dist/*.gz
if [ -e "build" ];then
rm -rf build
fi

pip install -r requirements.txt

MAX_JOBS=32 python setup.py sdist
echo "MAX_JOBS-------------"
MAX_JOBS=32 pip install dist/*.gz
