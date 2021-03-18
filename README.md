# Learning Attribute-Structure Co-Evolutions in Dynamic Graphs
**Description**: This repository contains the reference implementation of *CoEvoGNN* model proposed in the paper [Learning Attribute-Structure Co-Evolutions in Dynamic Graphs](https://arxiv.org/pdf/2007.13004.pdf) accepted by [The Second International Workshop on Deep Learning on Graphs: Methods and Applications (DLG-KDD’20)](https://deep-learning-graphs.bitbucket.io/dlg-kdd20/) and won the **Best Paper Award**.

## Usage
### 1. Dependencies
This code package was developed and tested with Python 3.7 and [PyTorch 1.0.1.post2](https://pytorch.org/).
Make sure all dependencies specified in the `./requirements.txt` file are satisfied before running the model. This can be achieved by
```
pip install -r requirements.txt
```
Other environment management tool such as [Conda](https://www.anaconda.com/) can also be used.

### 2. Data
The `./data/` folder contains the evolutionary co-authorship graphs (2k and 10k) descripted in the paper. Predefined paths for locating necessary data files can be found in the `./config.py` file.

**Note: Due to file size constraints, the 10k dataset files are compressed inside `./data/data_10k.zip`. Please unzip the file for training on the dataset. **

### 3. Run
To train the model, run
```
python main.py --dataset 2k --num_epochs 10
```
List of arguments:
+ `--dataset`: The dataset of evolutionary co-authorship graphs to use. Valid choices include `2k` and `10k`. Default is `2k`
+ `--t_0`: The start index of available time points. Default is `0`
+ `--T`: Length of available training time points. Default is `8`
+ `--t_train`: Length of training time points (from `--t_0`). Default is `8`
+ `--t_forecast`: Number of forecasting snapshots. Default is `1`
+ `--K`: Num of layers fusing new time point. Default is `2`
+ `--epochs`: Number of epochs for training. Default is `10`
+ `--H_0_npf`: File for initializing H_0.

## Examples
Other examples are provided in the `./example.sh` file.

## Miscellaneous
If you find this code pacakage is helpful, please consider cite us:
```
@article{wang2020learning,
  title={Learning Attribute-Structure Co-Evolutions in Dynamic Graphs},
  author={Wang, Daheng and Zhang, Zhihan and Ma, Yihong and Zhao, Tong and Jiang, Tianwen and Chawla, Nitesh V and Jiang, Meng},
  journal={arXiv preprint arXiv:2007.13004},
  year={2020}
}
```

