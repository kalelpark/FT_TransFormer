# Research on Tabular Deep Learning

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


**For paper implementations, see the section ["Papers and projects"](#papers-and-projects).**

**Experiment Model [FT Transformer, ResNet]**

```
$cd Researh
$sh experiment.sh 
```

## Papers and projects

| Name                                                          | Location                                                        | Comment        |
| :------------------------------------------------------------ | :-------------------------------------------------------------- | :------------- |
| Revisiting Pretrarining Objectives for Tabular Deep Learning  | [link](https://github.com/puhsu/tabular-dl-pretrain-objectives) | arXiv 2022     |
| On Embeddings for Numerical Features in Tabular Deep Learning | [link](https://github.com/Yura52/tabular-dl-num-embeddings)     | arXiv 2022     |
| `rtdl`                                                        | [link](https://github.com/Yura52/rtdl)                          | Python package |
