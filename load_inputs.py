from main import read_config
import util.dataloader as dataloader

if __name__ == '__main__':
    cfg = read_config()
    dataloader.load_data(config=cfg)
