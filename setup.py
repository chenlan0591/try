from setuptools import setup

setup(
    name='gym_beamhopping',
    version='0.0.1',
    description='beamhopping domain OpenAI Gym environment',
    author='Chenlan Lin',
    packages=['gym_beamhopping'],
    install_requires=['gym',
                      'numpy',  #'numpy>=1.14.0'
    ]
) 
