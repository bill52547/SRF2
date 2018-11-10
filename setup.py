from setuptools import setup

setup(name = 'SRF2',
      version = '0.0.1',
      description = 'Scalable Reconstruction Framework, version 2',
      license = 'LGPL',
      package_dir = {'': 'srf2'},
      install_requires = [
          'dxl-learn==0.2.1',
          'dxl-core==0.1.7',
          'doufo==0.0.4',
          'dxl-shape==0.1.2',
          'jfs==0.1.3',
          'scipy',
          'matplotlib',
          'typing',
          'h5py',
          'click',
          'jinja2',
          'pathlib',
          'numpy',
          'tqdm',
      ],
      zip_safe = False)
