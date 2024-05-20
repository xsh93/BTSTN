from setuptools import setup

setup(
    name='btstn',
    version='1.0',
    description='Bidirectional Time-series State Transfer Network (BTSTN)',
    author='',
    author_email='',
    packages=["btstn"],
    install_requires=[
        'torch==2.0.0',
        'numpy>=1.23.5',
        'pandas>=1.5.2',
        'pypots==0.5'
    ]
)
