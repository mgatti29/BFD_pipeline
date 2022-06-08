from __future__ import print_function
import sys,os,glob,re
import select

#from distutils.core import setup
from setuptools import setup
import distutils

print('Python version = ',sys.version)
py_version = "%d.%d"%sys.version_info[0:2]  # we check things based on the major.minor version.

dependencies = ['numpy', 'future', 'astropy', 'scipy']

version_file=os.path.join('metacal_m/','_version.py')
verstrline = open(version_file, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    ma_version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (version_file,))
print('MA version is %s'%(ma_version))

# data = glob.glob(os.path.join('data','*'))

dist = setup(
        name="metacal_m",
        version=ma_version,
        #author="Marco Gatti",
        #author_email="marcogatti29@gmail.com",
        description="Python module for measuring and modelling moments of convergence field maps",
#        long_description=long_description,
        license = "BSD License",
        #url="https://github.com/mgatti29/Moments_analysis",
        download_url="https://github.com/mgatti29/Moments_analysis/releases/tag/v%s.zip"%ma_version,
        packages=['metacal_m'],
#        package_data={'bfd' : data },
        install_requires=dependencies,
    )
