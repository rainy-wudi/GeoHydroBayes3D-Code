# Code availability section

Name of the code/library: GeoHydroBayes3D (version 2.0.0).

Contact: Dr. Chunhui Ma; e-mail: shanximachunhui@foxmail.com; phone number: +86 18700917478.

Hardware requirements: A 64-bit computer; CPU execution is supported and a GPU is optional. The Python workflow was tested on an Intel Core i7-14700KF computer with 32 GB RAM; this configuration is not a benchmarked minimum requirement.

Program language: Python and MATLAB.

Software required: Python 3.12 with NumPy, pandas, SciPy, scikit-learn, joblib, PyTorch and emcee (pinned versions are provided in requirements.txt). MATLAB with Statistics and Machine Learning Toolbox and a licensed Abaqus installation are additionally required to regenerate the finite-element database; the original solver log identifies Abaqus/Standard 2024.

Program size: 219,415 bytes of Python and MATLAB source code (approximately 0.22 MB), excluding datasets, trained models and documentation.

The source codes are available for downloading at the link: https://github.com/rainy-wudi/GeoHydroBayes3D-Code. The repository includes the processed numerical simulation database, hydraulic-head observations, pretrained models, installation instructions and runnable examples. Large Abaqus ODB files are not included.
