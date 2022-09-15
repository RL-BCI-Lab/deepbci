import yaml

def load_config(file):
    with open(file, 'r') as stream:
        try:
            params = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params

config_file = "config.yml"
config = load_config(config_file)
print(config)
