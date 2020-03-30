from setuptools import setup

setup(name='bento',
      version='0.1',
      description='Spatial RNA analysis toolkit',
      url='http://github.com/ckmah/benot',
      author='Clarence Mah',
      author_email='ckmah@ucsd.edu',
      license='MIT',
      packages=['bento'],
      install_requires=['pandas>=1.0.3'],
      zip_safe=False)
