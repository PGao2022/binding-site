from setuptools import setup

setup(
    name='tfbio',
    description='Tensorflow utilities for biological data',
    version='0.3',
    keywords=['neural networks', 'deep learning', 'bioinformatics', 'cheminformatics'],
    author='Marta M. Stepniewska-Dziubinska',
    author_email='martasd@ibb.waw.pl',
    url='http://gitlab.com/cheminfIBB/tfbio',
    license='BSD',
    packages=['tfbio',
              'tests'],
    test_suite='tests',
    package_data={'tests': ['tests/data']}
)
