import json
import os
from util.IMUDataset import IMUDataset


def read_config():
    __location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
     # Read configuration
    with open(os.path.join(__location__, 'config.json'), "r") as read_file:
        config = json.load(read_file)
    return config


def main():
    cfg = read_config()
    create_IMUDataset(cfg)


def create_IMUDataset(config):
    df = IMUDataset(config)
    print('[INFO] -- IMU-shape = {}'.format(df.imu.shape))
    print('[INFO] -- Labels-shape = {}'.format(df.labels.shape))

    print(df[0])
    print(len(df))


if __name__ == '__main__':
    main()
