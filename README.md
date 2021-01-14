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
Use `git status` to list all new or modified files that haven't yet been committed
