import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7, 8, 9' 
import yaml
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type = "str", required = True)
    args = parser.parse_args()
    
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with open("run.yaml") as f:
        config = yaml.load(f, Loader = yaml.FullLoader)[args.action]



    
    

    os.path.exists()