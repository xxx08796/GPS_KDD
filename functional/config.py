import os.path as osp
import json


def load_config(source, local_dir='./config'):
    path = osp.join(local_dir, source)
    if path.endswith('.json'):
        loaded = parse_json(path)
    else:
        raise NotImplementedError

    return loaded


def parse_json(path):
    content = open(path).read()
    return json.loads(content)
