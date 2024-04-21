# Spatiotemporal

### Contents 
This repository will share a methodology on modeling spatiotemporal data via deep learning methods. Specifically this is a recreation of Amato's (2020) framework for spatio-temporal prediction - https://www.nature.com/articles/s41598-020-79148-7.

### Methodology
The method from the paper decomposes the spatial and temporal data via EOF then uses a neural network focus primarily on the spatial features. The reconstruction layer is basically a model prediction using spatial covariates for the spatial components and full data is reformed via matrix algebra.  

### Setting up the environment
The package management is done by poetry. 
```
conda create -n st
conda activate st
conda install pip
pip install poetry
poetry install
# pip install tensorflow # This might be needed depending on your system.
# pip install tensorflow-macos # If using M chip
# pip install tensorflow-metal # If using M chip
jupyter notebook # Use the Run Spatial Analysis notebook
```

### Pending Work
- Apply to non-simulated data. 

### Output and viewing the results
Run the un Spatial Analysis notebook.

### Contact 
If you would like to collaborate or need any help with the code you can reach me at satesh.ramdhani@gmail.com. 