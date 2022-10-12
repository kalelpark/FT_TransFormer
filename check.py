import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

X_train, Y_train = np.load("data/aloi/N_train.npy"), np.load("data/aloi/y_train.npy")
print(np.shape(X_train), np.shape(Y_train))
X_split_train, X_split_valid, y_split_train, Y_split_valid = train_test_split(X_train, Y_train, test_size = 0.15)
print(np.shape(X_split_train), np.shape(X_split_valid), np.shape(y_split_train), np.shape(Y_split_valid))

# data_paths = "data"
# data_folders = os.listdir("data")


# # with open("run.yaml") as f:
# #         config = yaml.load(f, Loader = yaml.FullLoader)
# #         print(config["train"])
# #         print(config["path"])


# for data_folder in data_folders:
#     data_path = os.path.join(data_paths, data_folder)
#     os.path_join(data_path, "N_train")
#     y_train_path = os.path_join(data_path, "N_val")
#     y_train_path = os.path_join(data_path, "y_train")
#     print(data_path)
#     break


# trai