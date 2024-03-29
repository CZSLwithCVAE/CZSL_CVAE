# Adversarial Training of Variational Auto-encoders for Continual Zero-shot Learning(A-CZSL)
This is the official PyTorch implementation of [Adversarial Training of Variational Auto-encoders for Continual Zero-shot Learning(A-CZSL)](https://arxiv.org/abs/2102.03778). The paper has been accepted at IJCNN, 2021(oral).
# Abstract 
Most of the existing artificial neural networks(ANNs) fail to learn continually due to catastrophic forgetting, while humans can do the same by maintaining previous tasks' performances. Although storing all the previous data can alleviate the problem, it takes a large memory, infeasible in real-world utilization. We propose a continual zero-shot learning model(A-CZSL) that is more suitable in real-case scenarios to address the issue that can learn sequentially and distinguish classes the model has not seen during training. Further, to enhance the reliability, we develop A-CZSL for a single head continual learning setting where task identity is revealed during the training
process but not during the testing. We present a hybrid network that consists of a shared VAE module to hold information of all tasks and task-specific private VAE modules for each task. The model's size grows with each task to prevent catastrophic forgetting of task-specific skills, and it includes a replay approach to preserve shared skills. We demonstrate our hybrid model outperforms the baselines and is effective on several datasets, i.e., CUB, AWA1, AWA2, and aPY. We show our method is superior in class sequentially learning with ZSL(Zero-Shot Learning) and GZSL(Generalized Zero-Shot Learning).

# Authors:
[Subhankar Ghosh](https://sites.google.com/view/subhankarghosh/home)(Indian Institute of Science)
# Citation
If using this code, parts of it, or developments from it, please cite our paper:
```
@article{ghosh2021adversarial,
  title={Adversarial Training of Variational Auto-encoders for Continual Zero-shot Learning},
  author={Ghosh, Subhankar},
  journal={arXiv preprint arXiv:2102.03778},
  year={2021}
}
```
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
# Questions/ Bugs
For questions and bugs, please contact the author Subhankar Ghosh via email [subhankarg@alum.iis.ac.in](mailto:x@x.com)
# License
This source code is released under The MIT License found in the [LICENSE](https://github.com/CZSLwithCVAE/CZSL_CVAE/blob/main/LICENSE) file in the root directory of this source tree.
