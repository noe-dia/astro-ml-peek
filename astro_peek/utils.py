import yaml

def load_yaml(path): 
    with open(path) as file: 
        dict_yaml = yaml.safe_load(file)
    return dict_yaml