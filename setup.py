from setuptools import setup, find_packages

setup(
    name='ml-advanced',
    description='Advanced Machine Learning',
    author='gromak',
    install_requires=[],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'lab1 = Lab1.lab1:main',
        ]
    },
)
