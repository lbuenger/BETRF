from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy

# define an extension that will be cythonized and compiled
ext = Extension(name="hello", sources=["hello.pyx"], include_dirs=[numpy.get_include()], extra_compile_args=["-O3"])
setup(ext_modules=cythonize(ext))
