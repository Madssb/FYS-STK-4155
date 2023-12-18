# Project 3 - Classification of Clouds

Filenames pertaining to Project 3 are stored in this directory.

To reproduce the project, you'll need the following data unzipped in a directory with cloud-related data. You can obtain it by downloading and unzipping the dataset from the following source:

**Dataset Download Link:** [CCSN Dataset](https://www.kaggle.com/datasets/mmichelli/cirrus-cumulus-stratus-nimbus-ccsn-database/download?datasetVersionNumber=1)

Before running the project, please ensure that the required directory containing the data exists within this project directory. You can use the following terminal command to check for its existence:

```bash
if [ -d "CCSN_v2" ]; then
    echo "The directory 'CCSN_v2' exists. You can proceed with the project."
else
    echo "The directory 'CCSN_v2' does not exist. Please download and unzip the dataset into this directory before running the project."
fi
```

## File overview
To initialize and train a CNN model, run one of the following programs:
- cnn_cloudnet.py: Trains and saves a model on the CCSN dataset using input image size 227x227 pixels
- cnn_cloudnet128.py: Trains and saves a model on the CCSN dataset using input image size 128x128 pixels
- cnn_cloudnet28.py: Trains and saves a model on the CCSN dataset using input image size 28x28 pixels
- cnn_mnist.py: Trains and saves a model on the MNIST dataset

To visualize the model performance, run one of the following programs:
- cnn_visualization.py: Computes evaluations and creates plots cloud classification models (based on the CCSN dataset)
- cnn_visalization_mnist.py: Computes evaluations and creates plots of MNIST classification models
