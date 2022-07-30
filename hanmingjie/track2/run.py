import sys
import os
sys.path.append(os.path.abspath('..'))
# sys.path.append('/root/autodl-nas/ccac/hanmingjie')
import warnings
warnings.filterwarnings("ignore")
import time
import argparse

from common.yml_loader import YmlLoader
from track2.util.trainer import Trainer
from common.logger import Logger


def main():
    args = arg_parser()
    loader = YmlLoader({'yml_path': args.config})
    config = loader()
    init_logger(config)
    Logger.get_logger().info('-----------  start train  ------------')
    trainer = Trainer(config)
    trainer()
    Logger.get_logger().info('-----------  end of  train  ------------')


def arg_parser():
    parser = argparse.ArgumentParser(description="ccac")
    parser.add_argument('-config',help='the path of yml file', required=True)
    args = parser.parse_args()
    return args


def init_logger(config):
    log_dir = config['run_param']['log_dir']
    # if the dir is not existing, create it
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    log_filename = log_dir + time.strftime('%Y-%m-%d %H_%M_%S') + '.log'
    log_level = config['run_param']['log_level']
    Logger.set_config(log_filename, log_level)


if __name__ == '__main__':
    main()