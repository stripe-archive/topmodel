#!/usr/bin/env python

from setuptools import setup

setup(name='topmodel',
      version='0.1.2',
      description='Model evaluation',
      author='Julia Evans',
      author_email='julia@stripe.com',
      install_requires=['pandas==0.13.0', 'boto==2.25.0'],
      packages=['topmodel'],
      )
