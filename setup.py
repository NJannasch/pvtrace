from setuptools import setup, find_packages
from os import path
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
   name='pvtrace',
   version='2.0.4',
   description='Optical ray tracing for luminescent materials and spectral converter photovoltaic devices.',
   long_description=long_description,
   long_description_content_type='text/markdown',
   author='Daniel Farrell',
   author_email='dan@excitonlabs.com',
   url='https://github.com/danieljfarrell/pvtrace',
   download_url = 'https://github.com/danieljfarrell/pvtrace/archive/v2.0.4.tar.gz',
   python_requires='>=3.7.2',
   packages=find_packages(),
   keywords=[
       "optics",
       "simulation",
       "ray tracing",
       "photovoltaics",
       "solar",
       "energy",
       "nonimaging",
       "luminescence",
       "spectroscopy"
   ],
   install_requires=[
       "numpy",
       "scipy",
       "pandas",
       "anytree",
       "meshcat>=0.0.16",
       "trimesh[easy]"
   ],
   classifiers=[
     'Development Status :: 4 - Beta',
     'Intended Audience :: Science/Research',
     'Intended Audience :: Developers',   
     'Topic :: Scientific/Engineering :: Physics',
     'Topic :: Scientific/Engineering :: Chemistry',
     'Topic :: Scientific/Engineering :: Visualization',
     'License :: OSI Approved :: BSD License',
     'Programming Language :: Python :: 3.7'
   ]
)
