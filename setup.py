from setuptools import setup

with open("autofolio/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

setup(
    name='autofolio',
    version=version,
    packages=['autofolio', 'autofolio.io', 'autofolio.aslib', 'autofolio.facade', 'autofolio.selector',
              'autofolio.selector.regressors', 'autofolio.selector.classifiers', 'autofolio.validation',
              'autofolio.pre_solving', 'autofolio.feature_preprocessing'],
    url='',
    license="2-clause BSD",
    author="Marius Lindauer",
    author_email="lindauer@cs.uni-freiburg.de",
    description=("AutoFolio 2, an automaticalliy configured algorithm selector."),
    install_requires=[
        'Cython',
        'numpy',
        'scipy',
        'scikit-learn>=0.20.0',
        'matplotlib',
        'pandas',
        'xgboost',
        'ConfigSpace==0.6.1',
        'pyrfr',
        'smac',
        'clingo',
        'func_timeout',
        'liac-arff'
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: 2-clause BSD",
    ]
)
