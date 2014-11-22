import yaml

def read_config(filename):
    with open(filename) as f:
        config = yaml.load(f)
    if set(config.keys()) != set(["aws_access_key", "aws_secret_key", "bucket"]):
        raise Exception('Invalid config file: expecting keys "aws_access_key", "aws_secret_key", "bucket"')
    return config
