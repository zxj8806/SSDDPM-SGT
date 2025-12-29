This repository includes the source code of "**Biologically-Inspired Sparse-Spike Diffusion Modeling for Effective Representation Learning on Graphs**"

## 🗻 Install:

require: python 3.8+, pytorch and some common packages.

```
conda create -n py38 python=3.8
conda activate py38
pip install graphgallery pandas
pip install spikingjelly
pip install thop scikit-learn
```

<br/>

- In case are prompted that other dependent packages are missing, can install it with: pip install xxx.
- Set parameters in models_conf.json, such as device": "cuda:0"
  
  <br/>

## 🏝️ **Run**

```
cd path_to/handcode/
python run_snn.py
```




This project is motivated by [GraphGallery](https://github.com/EdisonLeeeee/GraphGallery.git), [spikingjelly](https://github.com/fangwei123456/spikingjelly.git) and [SpikingGCN](https://github.com/ZulunZhu/SpikingGCN), etc., and the original implementations of the authors, thanks for their excellent works!
