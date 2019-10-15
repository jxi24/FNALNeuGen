#!/usr/bin/env python
""" Setup script for the nuchic neutrino event generator. """

# from setuptools import setup, find_packages, Extension
from os import path
from io import open
import setuptools
from numpy.distutils.core import setup, Extension
# io.open is needed for projects that support Python 2.7
# It ensures open() defaults to text mode with universal newlines,
# and accepts an argument to specify the text encoding


def hadronic_tensor(filename):
    """ Get path to hadronic_tensor code. """
    return path.join('nuchic', 'hadronic_tensor', filename)


FORM_FACTOR = Extension(name='_form_factor',
                        sources=[hadronic_tensor('form_factor.cc'),
                                 hadronic_tensor('form_factor_wrap.cxx')],
                        extra_compile_args=['-std=c++11'],
                        )

NUCLEAR_RESPONSE = Extension(name='hadronic',
                             sources=[hadronic_tensor('hadronic.pyf'),
                                      hadronic_tensor('currents_opt_v1.f90'),
                                      hadronic_tensor('mathtool.f90'),
                                      hadronic_tensor('hadronic.f90')],
                             )

HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='nuchic',
    version='1.0',
    description='Neutrino Event Generator',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/jxi24/FNALNeuGen',
    author='Joshua Isaacson, \
            William Jay, \
            Alessandro Lovato, \
            Pedro A. Machado, \
            Stefan Prestel, \
            Noemi Rocco, \
            Holger Schulz',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.5, <4',
    install_requires=[
        'numpy',
        'vegas',
        'h5py',
        'scipy',
        'pandas',
        'sklearn',
        'absl-py',
        'matplotlib',
        'tqdm',
        'pyyaml',
    ],
    # Provide executable script to run the main code
    entry_points={'console_scripts': ['nuchic = nuchic.main:nu_chic', ], },
    ext_modules=[FORM_FACTOR, NUCLEAR_RESPONSE],
    package_data={'': ['data/*', 'data/qe/*', 'pke/*', 'configurations/*',
                       'template.yml']},
    extras_require={
        'test': ['pytest', 'coverage', 'pytest-cov'],
    },
)
