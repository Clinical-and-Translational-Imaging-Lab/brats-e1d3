import os
import yaml


def parse_yaml_config(file_path):
    assert '.yaml' in file_path
    assert os.path.exists(file_path)

    with open(file_path, "r") as stream:
        try:
            parsed_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_data
