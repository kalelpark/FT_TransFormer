import os
from torch import Tensor 
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
import json
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

number = 123
task_type = "binary"
d_out= 1 if task_type == "regression" else 109 if task_type == "multiclass" else 2
print(d_out)
# with open("data/aloi/info.json", "r") as f:
#         info_dict = json.load(f)

# with open("model.yaml") as f:
#         config = yaml.load(f, Loader = yaml.FullLoader)
#         config["fttransformer"]["valid_score"] = 123
#         config["fttransformer"]["ttrain_score"] = 123
#         config["fttransformer"]["test_score"] = 123
#         with open("tetetete", "w", encoding = "utf-8") as make_file:
#                 json.dump(config["fttransformer"], make_file, ensure_ascii = False, indent = "\t")


# info_dict = dict()
# info_dict["wongi"] = "helllooo"

# print(info_dict)

# with open("qqqqq/temp", "w", encoding = "utf-8") as make_file:
#         json.dump(info_dict, make_file, ensure_ascii = False, indent = "\t")

# temp = np.array(1)
# temp
# print(temp)
# a = np.array([1, 2, 3])
# print(np.shape(a))

# print

# class npy_dataset(Dataset):
#     def __init__(self, data, label):
#         self.data = data
#         self.label = label
    
#     def __getitem__(self, idx):
#         x_data = Tensor(self.data)[idx]
#         y_label = Tensor(self.label)[idx]

#         return x_data, y_label
    
#     def __len__(self):
#         return len(self.data)

# train_data, label_data = np.load("data/aloi/N_train.npy"), np.load("data/aloi/y_train.npy") 
# # print(np.shape(train_data), np.shape(label_data))
# custom_dataset = npy_dataset(data = train_data, label = label_data)
# train_dataloader = DataLoader(custom_dataset, batch_size = int(224), pin_memory = True)
# for i, t in train_dataloader:
#     print(i.size(), t.size())
# temp, temp_1 = next(iter(custom_dataset))
# print(Tensor(temp).size(), Tensor(temp_1).size())

# print(np.shape(train_data), np.shape(train_data[[1, 2, 3]]))





# for data_folder in data_folders:
#     data_path = os.path.join(data_paths, data_folder)
#     os.path_join(data_path, "N_train")
#     y_train_path = os.path_join(data_path, "N_val")
#     y_train_path = os.path_join(data_path, "y_train")
#     print(data_path)
#     break


# trai
