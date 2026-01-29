# code from Dr. Michael Weber (TROPOS)
from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('process_statsmodels.pyx'))


