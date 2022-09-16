# Set up to do builds and run using Intel-compiled programs on NERSC
module load cray-fftw
export OMP_NUM_THREADS=32
export cfitsio_DIR=${HOME}/PERLMUTTER_bfd

export FFTW_DIR=$FFTW_ROOT
export EIGEN_DIR=/global/u2/m/mgatti/eigen/
export CFITSIO_DIR=$cfitsio_DIR/
export GBUTIL_DIR=/global/u2/m/mgatti/gbutil/
export GBFITS_DIR=/global/u2/m/mgatti/gbfits/
#export YAML_DIR=/global/homes/m/mgatti/yaml-cpp/
export CXX="g++ -fopenmp"
export CC="gcc -fopenmp"
export CXXFLAGS="-O"
export CFLAGS="-O"
export LD_LIBRARY_PATH=$FFTW_DIR/lib:$CFITSIO_DIR/lib:$LD_LIBRARY_PATH

path=($HOME/INTEL/bin $HOME/CODE/bfd/bin $path)
export PATH

# Set up for MKL
#export unset MKL_LINK
#export unset MKL_OPTS
#export unset MKL_DIR
