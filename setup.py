from setuptools import setup, find_packages

setup(
    name='AutoGrad',
    version='0.0.1',
    url='https://github.com/mypackage.git',
    author='Michael Pearce',
    author_email='scrambledpie@gmail.com',
    description='Toy autograd and MLP trainer for MNIST',
    packages=find_packages("."),
    install_requires=['numpy >= 1.26'],
)