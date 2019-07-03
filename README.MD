# Diffuse retro-reflective imaging for improved mosquito tracking around human baited bednets
## Data and code release

This repository contains all code and data (with links to other data store where relevant), which was used to generate almost all images and result metrics in the paper

Vitaly Voloshin, Christian Kröner, Chandrabhan Seniya, Gregory P.D. Murray, Amy Guy, Catherine E. Towers, Philip J. McCall, and David P. Towers. Diffuse retro-reflective imaging for improved mosquito tracking around human baited bednets. bioRxiv doi: 

I would recommend you to read through the paper in order to understand what this code is all about.

## Code

This repository contains two Jupyter Notebooks:

- `illuminationAnalysis.ipynb` - a notebook, which contains all the code that was used to generate all images related to image analysis
- `calcUndetectedNew.ipynb` - a notebook, which contains all code, which was used to generate all performance metrics presented in Table 1 of the paper.

## Data

All data necessary for the code to work is available in this repository except sequence files (which should be placed into the `IlluminationData` folder), which are available at Figshare repository (doi: 10.6084/m9.figshare.8534057).

- There are two StreamPix Sequence v.5 files, each of which contains 500 frames for illumination analysis: `backlit.seq` (recorded on the backlit system) and `reducedRRS.seq` (recorded on retroreflective screen system), which are available at Figshare (doi: 10.6084/m9.figshare.8534057). The files should be placed into the `IlluminationData` folder for the Jupyter Notebook `illuminationAnalysis.ipynb` to work.
- There are 1 frame sequence file (for background image only), positions and track files for gaps analysis and performance metrics generation. They are located in `GapsData` folder in this repository. All data related to backlit system is available in `BackLit` folder and all data related to retroreflective screen system is available in `RRS` folder.

## Installation

In order to install it just clone this repository into your desired folder:

```
git clone https://github.com/Pigrenok/TrackingPerformance.git
```

You also need to have Python 3.6+ installed and have all the dependencies. In order to install all dependencies, you can use `requirements.txt` file provided:

```
pip install -r requirements.txt
```

This ideally needs to be done in the local Python environment. For managing environment, you can use, for example, [`pyenv` project](https://github.com/pyenv/pyenv). 

## Usage

In order to use it, you need to start local Jupyter server or Jupyter Lab server and then in the Jupyter interface open any one of the Jupyter Notebooks provided. The notebooks contain importing all required libraries, all necessary functions and also cells that actually generate all the images. Each function contains short description as a Python Docstring and the usage cells should be self-explanatory.

## Issues

If you have any issues with this code, please, raise the issue here, on Github and I will try to get back to you as soon as possible.
