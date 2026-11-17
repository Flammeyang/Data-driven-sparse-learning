  A data-driven method is proposed for reconstructing 3D coastal subsurface geological model. Neighborhood graphs combined with depth-wise sequential modeling are used to capture soil spatial correlations. The method improves the accuracy of soil property prediction compared to the graph neural network method alone. The method is sparse data driven and does not require 2D/3D training images from similar sites.

# Validated Windows environment: Python 3.8.20, CUDA 11.7, cuDNN 8.5.0.
# Use the matching PyTorch build for the selected CPU/CUDA environment.
torch==1.13.0
numpy==1.24.4
pandas==2.0.3
scipy==1.10.1
openpyxl==3.1.5
libpysal==4.8.1
geopandas==0.13.2
shapely==2.0.7
scikit-learn==1.3.2
Seed==80

