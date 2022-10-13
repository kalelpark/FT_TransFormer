import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '8, 9' 
import yaml
import typing as ty
import argparse
import torch
from train import model_train
from infer import model_infer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type = "str", required = True)
    args = parser.parse_args()
    
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with open("run.yaml") as f:
        config = yaml.load(f, Loader = yaml.FullLoader)[args.action]
    if args.action == "train":
        model_train(args, config)
    else:
        model_infer(args, config)