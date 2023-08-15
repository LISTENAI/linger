from setuptools import find_packages, setup

setup(
    name="pylinger",
    version="1.1.1",
    description="linger is package of fix training",
    author="listenai",
	author_email="lingerthinker@listenai.com",
	url="https://github.com/LISTENAI/linger",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Programming Language :: C++',
            'Programming Language :: Python :: 3',
    ]
)
