# Set up to do builds and run using Intel-compiled programs on NERSC
module swap PrgEnv-intel
module load cray-fftw
export OMP_NUM_THREADS=32
export INTEL_DIR=${HOME}/INTEL

export FFTW_DIR=$FFTW_ROOT
export EIGEN_DIR=/global/u2/m/mgatti/eigen/
export CFITSIO_DIR=$INTEL_DIR/cfitsio
export GBUTIL_DIR=/global/u2/m/mgatti/gbutil/
export GBFITS_DIR=/global/u2/m/mgatti/gbfits/
export YAML_DIR=/global/homes/m/mgatti/yaml-cpp/
export CXX="icpc -qopenmp -mkl"
export CC="icc -qopenmp"
export CXXFLAGS="-O -mkl"
export CFLAGS="-O -mkl"
export LD_LIBRARY_PATH=$FFTW_DIR/lib:$CFITSIO_DIR/lib:$LD_LIBRARY_PATH

path=($HOME/INTEL/bin $HOME/CODE/bfd/bin $path)
export PATH

# Set up for MKL
export MKL_LINK
export MKL_OPTS
export MKL_DIR
