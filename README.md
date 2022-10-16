# Research on Tabular Deep Learning

**This Project training 8 files at once as a Tabular Deep Learning model and stores Experimental results. additionaly use wandb.**

**For paper implementations, see the section ["Papers and projects"](#papers-and-projects).**

## Results
We saved test information in `"Output/[model]/data/default/info.json"`. Check it Out!

## Datasets

We upload the datasets used in the paper with our train/val/test splits [here](https://www.dropbox.com/s/cj9ex11u6ri0tdy/tabular-pretrains-data.tar?dl=1). We do not impose additional restrictions to the original dataset licenses, the sources of the data are listed in the paper appendix.

You could load the datasets with the following commands:

``` bash
conda activate tdl
cd $Researh
wget "https://www.dropbox.com/s/o53umyg6mn3zhxy/data.tar.gz?dl=1" -O rtdl_data.tar.gz
tar -zvf rtdl_data.tar.gz
```


**Setup the enviroment for train**

```bash
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
