# CZSL_CVAE


# Prerequisites:
- Linux-64
- Python 3.6
- PyTorch 1.3.1
- CPU or NVIDIA GPU + CUDA10 CuDNN7.5

# Installation
- Create a conda environment and install few packages:
 ```
conda create -n <env name> python = 3.6
conda activate <env name>
pip install -r requirements.txt
```
- Clone the repository:
 ```
mkdir CL_ZSL
cd CL_ZSL
git clone https://github.com/CZSLwithCVAE/CZSL_CVAE.git
```
- The following structure is expected in the main directory:
 ```
./data                          : It contains all four datasets, such as CUB, aPY, AWA1, and AWA2.
./networks                      : It has architectures of the discriminator, model, and classifier.
a_vae.py                        
classi.py
main.py
utils.py
```
For each dataset we need to download the data from the `data` directory and have to add the paths to `main.py` to run the code successfully.
Once the data paths are placed correctly, please run the following command.
```
python main.py
```

We are not claiming the used hyperparameters give the best results, but you are free to explore and let us know better results. 

# License
This source code is released under The MIT License found in the LICENSE file in the root directory of this source tree.
