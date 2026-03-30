from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 10, \
    "This patched version of Spinning Up requires Python 3.10 or greater."

with open(join("spinup", "version.py")) as version_file:
    exec(version_file.read())

setup(
    name='spinup',
    py_modules=['spinup'],
    version=__version__,#'0.1',
    install_requires=[
        'cloudpickle',
        'gymnasium[classic-control,box2d]>=1.2.0',
        'ipython',
        'joblib',
        'matplotlib',
        'mpi4py',
        'numpy',
        'pandas',
        'pytest',
        'psutil',
        'scipy',
        'seaborn',
        # 'tensorflow>=1.8.0,<2.0',
        'torch',
        'tqdm'
    ],
    description="Teaching tools for introducing people to deep RL.",
    author="Joshua Achiam",
)
