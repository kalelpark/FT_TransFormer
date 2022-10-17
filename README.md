# Research on Tabular Deep Learning

**This Project training 8 files at once as a Tabular Deep Learning model and stores Experimental results. additionaly use wandb.**

**For paper implementations, see the section ["Papers and projects"](#papers-and-projects).**

## Setup the enviroment for evaluation

```bash
$cd Researh
$sh experiment.sh 
```

## Single
Single train. in `"python main.py --action train --model fttransformer --data microsoft --savepath output"`.

## Results
We saved reult information in `"Output/model_name/data/default/info.json"`.

## Datasets

We upload the datasets used in the paper with our train/val/test splits [here](https://www.dropbox.com/s/cj9ex11u6ri0tdy/tabular-pretrains-data.tar?dl=1). We do not impose additional restrictions to the original dataset licenses, the sources of the data are listed in the paper appendix.

You could load the datasets with the following commands:

``` bash
conda activate tdl
cd $Researh
wget "https://www.dropbox.com/s/o53umyg6mn3zhxy/data.tar.gz?dl=1" -O rtdl_data.tar.gz
tar -zvf rtdl_data.tar.gz
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
| Revisiting Pretrarining Objectives for Tabular Deep Learning  | [link](https://arxiv.org/abs/2207.03208) | arXiv 2022     |
| On Embeddings for Numerical Features in Tabular Deep Learning | [link](https://arxiv.org/abs/2203.05556)     | arXiv 2022     |