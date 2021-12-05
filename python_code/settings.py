"""
Created on 21/06/2018
@author: Nicolai Pedersen
"""

import os


def path_init():
    data_path = data_path_init()
    # log_path = log_path_init()
    # experiment_data_path = experiment_data_init()
    # model_path = model_path_init()
    # return data_path, log_path, experiment_data_path, model_path
    return data_path


def base_dir_init():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def data_path_init():
    return os.path.normpath(base_dir_init() + '/data/')



if __name__ == '__main__':
    base_dir = base_dir_init()
    print('====== Current dir ======')
    print(base_dir)
