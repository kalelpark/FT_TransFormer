import yaml

with open("temp.yaml") as f:
    data = yaml.load(f, Loader = yaml.FullLoader)
    print(data)