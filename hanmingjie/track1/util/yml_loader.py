import yaml

class YmlLoader:
    def __init__(self, config):
        self.yml_path = config['yml_path']

    def __call__(self):
        with open(self.yml_path, encoding='utf-8') as f:
            content = f.read()
        config = yaml.load(content, Loader=yaml.SafeLoader)

        return config


