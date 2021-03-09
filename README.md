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
* run `python -m GraphGenerator --phase preprocessing -i google.txt -o google.graph`

**2. Train and infer new graphs**
* run `python -m GraphGenerator --phase train -i google.graph -o new_google.graphs -g vgae --config config/vgae.yaml`

**3. Evaluate the results**
* run `python -m GraphGenerator --phase evaluate -i new_google.graphs -r google.graph`
