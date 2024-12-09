import yaml
import argparse

class Config(argparse.Namespace):
    @classmethod
    def from_yaml(cls, file_path):
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls._dict_to_namespace(config_dict)

    @staticmethod
    def _dict_to_namespace(data):
        if isinstance(data, dict):
            return Config(**{k: Config._dict_to_namespace(v) for k, v in data.items()})
        elif isinstance(data, list):
            return [Config._dict_to_namespace(item) for item in data]
        else:
            return data