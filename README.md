# BFD_pipeline

This GitHub repository hosts a comprehensive codebase implementing the Bayesian Fourier Domain Method (https://arxiv.org/abs/1508.05655) for measuring weak lensing shear from a sample of galaxy stamps. The code is designed to handle both real and simulated data, offering flexibility in analyzing various datasets.

The primary focus of the code is the analysis of Dark Energy Survey Year 6 (DES Y6) data, providing powerful tools to extract valuable shear measurements from the observed galaxy stamps. However, the repository also includes functionalities to generate simulated data and perform corresponding analysis, making it a versatile tool applicable to a wide range of scenarios beyond DES Y6.



# install (use pip install . when possible)
recently tested with python 3.6.3
- pandas (1.1.5)
- pyfits (3.5)
- matplotlib (3.3.49)
- frogress (0.9.1)
- numba (0.53.1)
- scipy (1.5.4)
- ngmix (2.2.1) [https://github.com/esheldon/ngmix.git]
- galsim (2.4.5)
- LSSTDESC.Coord (1.2.3) [installed by galsim automatically]
- astropy (4.1) [installed by galsim automatically]
- pybind11 (2.10.1) [installed by galsim automatically]
- esutil (0.6.9)
- fitsio (1.1.8) [https://github.com/esheldon/fitsio] 
- joblib (1.1.1)
- bfd (0.1) [https://github.com/gbernstein/bfd/]
- psfex (0.4.0) [https://github.com/esheldon/psfex.git]
- shredder  (1.0.0) [https://github.com/esheldon/shredder.git] [ngmix2 branch]
- meds (0.9.16) [https://github.com/esheldon/meds.git]
- sxdes (0.3.0) [https://github.com/esheldon/sxdes.git]
- sep (1.2.1)
- pytest
- pyyaml (5.4.1)

For the cpp part, needed only for the Bayesian integration (and not the computation of moments):
module swap PrgEnv-Intel (on nersc!)


# How to run the code
The code can be run by using a particular configuration file, specifying the steps of the pipeline, as follows:
./run_bfd.py --config config/config_data.yaml.
The following config files allows one to simulate galaxies and measure their shears through using the BFD code:

- config_tile.yaml
- config_tile_1gal.yaml
- config_tile_blends.yaml
- config_tile_stamps.yaml

As for data, an example run is described in 

- config_data.yaml


 x DESDM folks: in order to process meds files for the des y6 targets, do
The config file needs to include the path to the meds files & shredder files.

to run:\n

./run_bfd.py --config config/config_data.yaml --tiles DES_TILES

where DES_tiles can be a list of string, e.g: DES0137-3749, DES0137-3721


# Contributing
Contributions to this project are welcome! If you encounter any issues, have ideas for improvements, or would like to add new features, please submit a pull request or open an issue on the GitHub repository.

# License
This project is licensed under the MIT License. See the LICENSE file for more information.


