from setuptools import setup, find_packages

setup(
    name='pyedautils',
    version='0.1.0',
    author='Reto Marek',
    description='Python Energy Data Analysis Utilities',
    url='https://github.com/retomarek/pyedautils',
    license='BSD 2-clause',
    packages=find_packages(exclude="tests"),  # Automatically find packages in the current directory
    install_requires=[],  # List of dependencies
)
