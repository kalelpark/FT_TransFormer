import os
import yaml

# data_paths = "data"
data_folders = os.listdir("data")

with open("run.yaml") as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
        print(config["train"])
        print(config["path"])


# for data_folder in data_folders:
#     data_path = os.path.join(data_paths, data_folder)
#     file = os.listdir(data_path)
#     print(file)
