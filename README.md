# Research on Tabular Deep Learning

**For paper implementations, see the section ["Papers and projects"](#papers-and-projects).**

## Results
You access `Output/Model/data/default/info.json`. This file show model, accuracy, rmse etc.. check
### Datasets

We upload the datasets used in the paper with our train/val/test splits [here](https://www.dropbox.com/s/cj9ex11u6ri0tdy/tabular-pretrains-data.tar?dl=1). We do not impose additional restrictions to the original dataset licenses, the sources of the data are listed in the paper appendix.

You could load the datasets with the following commands:

``` bash
conda activate tdl
cd $PROJECT_DIR
wget "https://www.dropbox.com/s/cj9ex11u6ri0tdy/tabular-pretrains-data.tar?dl=1" -O tabular-pretrains-data.tar
tar -xvf tabular-pretrains-data.tar
```



**Setup the enviroment for train**

```
$cd Researh
$sh experiment.sh 
```

**File Structure**
```
├── Data
│   ├── microsoft
│   │     └── ...
│   ├── yahoo
│   │     └── ...
│   └── etc..
├── Output
│   ├── ft-transformer
│   │     ├── microsoft
│   │     │     ├── default
│   │     │     └── ensemble
│   │     └── yahoo
│   │           └── etc..
│   └── resnet..
├── config.yaml     "Model Architecture parameters.."
├── experiment.sh
├── main.py
├── infer.py
├── train.py
├── model.py
├── utils.py
etc..
``` 

## Papers and projects

| Name                                                          | Location                                                        | Comment        |
| :------------------------------------------------------------ | :-------------------------------------------------------------- | :------------- |
| Revisiting Pretrarining Objectives for Tabular Deep Learning  | [link](https://github.com/puhsu/tabular-dl-pretrain-objectives) | arXiv 2022     |
| On Embeddings for Numerical Features in Tabular Deep Learning | [link](https://github.com/Yura52/tabular-dl-num-embeddings)     | arXiv 2022     |
| `rtdl`                                                        | [link](https://github.com/Yura52/rtdl)                          | Python package |
