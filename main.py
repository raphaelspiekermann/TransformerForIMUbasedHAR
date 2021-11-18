import util.dataloader as dataloader
import json
import os


def read_config():
    __location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
     # Read configuration
    with open(os.path.join(__location__, 'config.json'), "r") as read_file:
        config = json.load(read_file)
    return config

if __name__ == '__main__':
    cfg = read_config()
    dataloader.load_data(config=cfg)
