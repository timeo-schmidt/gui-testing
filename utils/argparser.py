import yaml

class Config:
    def __str__(self):
        return self._recursive_str(self, "")

    def _recursive_str(self, obj, indent):
        result = []
        indent += "  "
        for key, value in obj.__dict__.items():
            if isinstance(value, Config):
                result.append(f"{indent}{key}:\n{self._recursive_str(value, indent)}")
            else:
                result.append(f"{indent}{key}: {value}")
        return "\n".join(result)


def dict_to_obj(d, obj):
    for key, value in d.items():
        if isinstance(value, dict):
            sub_obj = Config()
            setattr(obj, key, sub_obj)
            dict_to_obj(value, sub_obj)
        else:
            setattr(obj, key, value)

def load_config_file(filepath="config.yaml"):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)

    cfg = Config()
    dict_to_obj(config, cfg)
    
    return cfg