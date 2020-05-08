from setuptools import setup

setup(name='bento',
      version='0.1',
      description='Spatial RNA analysis toolkit',
      url='http://github.com/ckmah/bento',
      author='Clarence Mah',
      author_email='ckmah@ucsd.edu',
      license='MIT',
      packages=['bento'],
      install_requires=['altair>=4.1.0',
                        'opencv-python-headless>=4.2.0.32',
                        'geopandas>=0.7.0',
                        'loguru>=0.4.1',
                        'matplotlib>=3.2.1',
                        'numpy>=1.18.4',
                        'pandarallel>=1.4.8',
                        'pandas>=1.0.3',
                        'scikit-learn>=0.22.2.post1'
                        'scipy>=1.4.1',
                        'shapely>=1.7.0',
                        'tqdm>=4.44.1',
                        'umap-learn>=0.3.10'],
      zip_safe=False)
