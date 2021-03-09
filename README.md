# GraphGenerator
Toolkit for generating graphs and evaluating graph generators.

## Installation
### Requirements
- CentOS==7.5
- CUDA==11.1
- Python==3.6.8
- PyTorch==1.8.0

**1. Install Pytorch**
```bash
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```
**2. Clone and install**
```bash
git clone https://github.com/xiangsheng1325/GraphGenerator.git
cd GraphGenerator
pip install -r requirements.txt
pip install -e .
```
**3. Test the installation**
```bash
bash run_test.sh
```

## GraphGenerator Usage

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

**3. Train and infer new graphs**

Enjoy your graph simulation and graph data generation.

_Example:_
* run `python -m GraphGenerator --phase train -i google.graph -o new_google.graphs -g vgae --config config/vgae.yaml`

**4. Evaluate the results**

Calculating the distance between two set of graphs to evaluate the experimental results. 

_Example:_
* run `python -m GraphGenerator --phase evaluate -i new_google.graphs -r google.graph`

