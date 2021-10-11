# GraphGenerator
[![CodeSize](https://img.shields.io/github/languages/code-size/xiangsheng1325/GraphGenerator?style=plastic)](https://github.com/xiangsheng1325/GraphGenerator)
[![Contributor](https://img.shields.io/github/contributors/xiangsheng1325/GraphGenerator?style=plastic&color=blue)](https://github.com/xiangsheng1325/GraphGenerator/graphs/contributors)
[![Activity](https://img.shields.io/github/commit-activity/m/xiangsheng1325/GraphGenerator?style=plastic)](https://github.com/xiangsheng1325/GraphGenerator/pulse)

Toolkit for simulating observed graphs, generating new graphs and evaluating graph generators.

## Installation
### Environments
[![Python](https://img.shields.io/badge/Python-v3.6.8-blue?style=plastic)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-v1.8.1-green?style=plastic)](https://pypi.org/project/torch/)
[![Tensorflow](https://img.shields.io/badge/Tensorflow-v2.4.0-blue?style=plastic)](https://pypi.org/project/tensorflow/)

If users want to use some deep learning based graph generators, deep learning dependencies are required such as Pytorch or Tensorflow.
We prefer to use PyTorch as dependency.

**1. Install Pytorch**
```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
**2. Clone and install**
```bash
git clone https://github.com/xiangsheng1325/GraphGenerator.git
cd GraphGenerator
pip install -r requirements.txt
pip install -e .
```
### Dependencies
Users need to install specific dependencies to support some graph generators, which is listed here:

|Graph Generator|Dependencies|Graph Generator|Dependencies|
|--|--|--|--|
|ARVGA|Tensorflow|GraphRNN|Pytorch|
|BiGG|Pytorch|MMSB|Tensorflow Edward|
|BTER|MATLAB|NetGAN|Tensorflow|
|CondGEN|Pytorch|SBMGNN|Tensorflow|
|GRAN|Pytorch|SGAE|Pytorch|
|Graphite|Pytorch/Tensorflow|VGAE|Pytorch|


### Project organization
This project is modularized to benefit further contributions on it.
Please organize this project according to following structure:

```
GraphGenerator/
|___GraphGenerator/  # source code
|   |___models/ # graph generator implementations
|   |   |___bigg/
|   |   |   |___tree_model.py
|   |   |   |___...
|   |   |___sbm.py
|   |   |___...
|   |___metrics/
|   |   |___mmd.py
|   |   |___...
|   |___train.py
|   |___...
|
|___setup.py 
|
|___config/  # detailed configurations of complex models
|   |___graphite.yaml
|   |___...
|
|___data/  # raw data / cooked data
|   |___google.txt
|   |___...
|
|___exp # trained model and generated graphs
|   |___VGAE/
|   |___...
|
|___...
```

## GraphGenerator Usage
Here are some examples of using this toolkit.

**1. Preprocess data**

We prefer to converting graph data into the same data type. If the input data is ready, this step can be skipped.

_Example:_
* run `python -m GraphGenerator --phase preprocessing -i google.txt -o google.graph`

**2. Test the usage of graph generator**

Before training the deep learning based graph generators,
we prefer to test whether there are bugs in our model implementations.
If the generator runs well, this step can be skipped.

_Example:_
* run `python -m GraphGenerator --phase test -g bigg --config config/bigg.yaml`

Note that some algorithms may be affected by the CUDA version. (For example, Bigg may encounter problems during testing,
please refer to [this page](https://github.com/xiangsheng1325/GraphGenerator/blob/main/GraphGenerator/models/bigg_ops/tree_clib/reame.md)
to find resolutions.)


**3. Train and infer new graphs**

Enjoy your graph simulation and graph data generation.

_Example:_
* run `python -m GraphGenerator --phase train -i google.graph -g vgae --config config/vgae.yaml`

**4. Evaluate the results**

Calculating the distance between two set of graphs to evaluate the experimental results. 

_Example:_
* run `python -m GraphGenerator --phase evaluate -i new_google.graphs -r google.graph`

# Reference
Please use the following BibTex to cite this work if it makes contributions to your publications.

BibTex:
```
@Article{Xiang2021General,
    author={Xiang, Sheng and Wen, Dong and Cheng, Dawei and Zhang, Ying and Qin, Lu and Qian, Zhengping and Lin, Xuemin},
    title={General Graph Generators: Experiments, Analyses, and Improvements},
    journal={The VLDB Journal},
    year={2021},
    month={Oct},
    day={07},
    issn={0949-877X},
    doi={10.1007/s00778-021-00701-5},
    url={https://doi.org/10.1007/s00778-021-00701-5}
}
```