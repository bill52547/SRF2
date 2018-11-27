from setuptools import setup

setup(name = 'SRF2',
      version = '0.0.1',
      description = 'Scalable Reconstruction Framework, version 2',
      license = 'LGPL',
      package_dir = {'': 'srf2'},
      install_requires = [
          'scipy',
          'matplotlib',
          'h5py',
          'click',
          'pathlib',
          'numpy',
          'tqdm',
      ],
      zip_safe = False)
