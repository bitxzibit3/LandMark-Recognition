import yaml


def read_yaml(path):
	with open(path, 'r') as f:
		config = yaml.safe_load(f)
	return config
