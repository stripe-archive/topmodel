#!/usr/bin/env python

from setuptools import setup

setup(name='topmodel',
      version='0.2.0',
      description='Model evaluation',
      author='Julia Evans',
      author_email='julia@stripe.com',
      install_requires=['pandas>=0.13.1',
                        'boto>=2.34.0',
                        'ipython>-1.0.0',
                        'matplotlib==1.4.0',
                        'numpy>=1.8.1',
                        'pyyaml>=3.10',
                        'scikit_learn>=0.14.1',
                        'scipy>=0.13.3',
                        'mpld3>=0.2'
                        ],
      packages=['topmodel'],
      )
