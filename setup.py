from setuptools import setup
from pip.req import parse_requirements

setup(
    name = 'Quandl',
    description='Package for Quandl API access',
    version='1.5.1',
    author=", ".join(['Mark Hartney','Chris Stevens','Alex Kestner']),
    maintainer='Alex Kestner',
    maintainer_email='alex.kestner@gmail.com',
    url='http://www.quandl.com/',
    license='MIT',
    install_requires=[str(ir.req) for ir in parse_requirements('requirements.txt')],
    packages=['Quandl'],
)
