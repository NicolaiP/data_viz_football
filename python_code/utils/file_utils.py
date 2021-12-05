from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import pickle
import numpy as np
import scipy.io as sio
import h5py
from functools import wraps
import json
import shutil
sys.path.append("../")  # go to parent dir
sys.path.append("../../")  # go to parent dir
from python_code import settings


def save_as_pickle(variable_name, save_name):
    """Saves variable as pickle file.
    # Arguments
        save_name: Name of file.
    # Example
        dataPath = "C:/Users/nicol/Desktop/Master/Data/"
        save_name = dataPath + 'predictionsResNet50ADAM_lr0001_decay0005'
        file_utils.save_as_pickle(preds, save_name)
    """
    f = open(save_name + '.pckl', 'wb')
    pickle.dump(variable_name, f)
    f.close()


def load_pickle_file(path):
    """Loads pickle file.
    # Arguments
        path: Path to file.
    # Returns
        var: Loaded variables.
    # Example
        dataPath = "C:/Users/nicol/Desktop/Master/Data/"
        fileName = dataPath + 'predictionsResNet50ADAM_lr0001_decay0005'
        var = file_utils.load_pickle_file(path)
    """
    if path.split('.')[-1] == 'pckl':
        var = pickle.load(open(path, 'rb'))
    else:
        var = pickle.load(open(path + '.pckl', 'rb'))
    return var


def saveVariableAsMat(variableName, fileName):
    """Saves variable as .mat file.
    # Arguments
        fileName: Name of file.
    # Example
        dataPath = "C:/Users/nicol/Desktop/Master/Data/"
        fileName = dataPath + 'predictionsResNet50ADAM_lr0001_decay0005'
        file_utils.saveVariableAsMat(preds, fileName)
    """
    sio.savemat(fileName + '.mat', {'data': variableName})


def load_mat_var(path):
    """Load a variable from .mat file.
    # Arguments
        fileName: Name of file.
    # Example
        path = "C:/Users/nicol/Desktop/Master/Data/predictionsResNet50ADAM_lr0001_decay0005.mat"
        data = file_utils.load_mat_var(path)
    """
    return sio.loadmat(path)


def folderFinder(directory):
    """Finds names of folders in a given directory
    # Arguments
        directory: Path to the data.
    # Returns
        classes: List of classes in the folder.
    # Example
        directory = dataPath + "Classes"
        classes = file_utils.folderFinder(directory)
    """
    found_folders = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            found_folders.append(subdir)
    return found_folders


def standardize_path(path):
    standard_path = os.path.normpath(path)
    standard_path = standard_path.replace('\\', '/')
    return standard_path


def file_finder(directory, file_type='jpg', only_name=False):
    """Finds specified file types of in all subfolders of a given directory
    # Arguments
        directory: Path to the data.
        file_type: string or tuple of strings specifying the file type
    # Returns
        found_files: List of files in the folders.
    # Example
        directory = dataPath + "Classes"
        found_files = file_utils.fileFinder(directory)
    """
    found_files = []
    directory = standardize_path(directory)
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(file_type):
                found_files.append(os.path.join(root, f))
    if only_name:
        found_files = [os.path.basename(ii).split('.')[0] for ii in found_files]

    return found_files


def txtSaver(path, variable):
    """Saves a numpy array to a txt file.
    # Arguments
        path: Path to the data.
        variable: name of the variable to save
    # Returns
        txtArray: List of the txt content
    # Example
        path = "C:/Users/nicol/Desktop/Master/Data/PreprocessedSophiesChoice/scenes.txt"
        file_utils.txtSaver(dataPath, y)
    """
    np.savetxt(path, variable)


def txtLoader(path):
    """Loads a txt file into a numpy array.
    # Arguments
        path: Path to the data.
    # Returns
        txtArray: List of the txt content
    # Example
        dataPath = "C:/Users/nicol/Desktop/Master/Data/"
        txtArray = file_utils.txtLoader(dataPath + 'ISC_sophie.txt')
    """
    txt_array = np.loadtxt(path)
    return txt_array


def jsonLoader(path):
    """Loads a json file into a dict.
    # Arguments
        path: Path to the data.
    # Returns
        jsonDict: List of the txt content
    # Example
        dataPath = 'C:/Users/nicol/.keras/models/imagenet_class_index.json'
        jsonDict = file_utils.jsonLoader(dataPath)
    """
    json_dict = json.load(open(path))
    return json_dict


def make_folder(data_path):
    '''
    Function that creates a folder if it doesn't exist
    :param data_path:
    :return:
    '''
    if not os.path.exists(data_path):
        os.makedirs(data_path)


def makeClassFolders(data_path, classes):
    """Creates folders if missing
    # Arguments
        dataPath: Path to the data.
        classes: List of classes.
        folderName: Name of the folder as a string.
    # Returns
        missingClasses: List of missing classes.
    # Example
        directory = dataPath + "Classes"
        classes = FindClasses(directory)
        trainingFolderName = 'train'
        missingTrain = file_utils.makeClassFolders(dataPath, classes, trainingFolderName)
    """
    for idxClass in classes:
        new_path = data_path + '/' + idxClass
        make_folder(new_path)


def getTimestamp():
    current_time = time.localtime()
    timestamp = str(current_time.tm_year) + '.' + str(current_time.tm_mon) + '.' + str(current_time.tm_mday) + '_' + str(current_time.tm_hour) + '.' \
                + str(current_time.tm_min)
    return timestamp


def take_time(input_time=False):
    current_time = time.time()
    if input_time:
        current_time = current_time - input_time
        print(current_time)
    return current_time


def increment_folder_number(log_dir, folder_name):
    current_log_folders = folderFinder(log_dir)
    if folder_name not in current_log_folders:
        log_dir = os.path.join(log_dir, folder_name + '1')
    else:
        new_number = str(int(current_log_folders[-1][-1]) + 1)
        log_dir = os.path.join(log_dir, folder_name + new_number)

    return log_dir


def copy_file_2_dir(file, dst_dir):
    """
    Be aware that it is also possible to change the name of the copied file
    :param file:
    :param dst_dir:
    :return:
    :Example:
    audio_path = 'C:/Users/nicol/Desktop/lrs3_trainval/trainval/'
    new_dir = 'C:/Users/nicol/Documents/GitHub/PhD/data/lrs3_trainval'
    movie_files = file_finder(audio_path, 'mp4')
    for file in movie_files:
        new_name = '_'.join(file.split('\\')[-2:])
        copy_file_2_dir(file, new_dir + '\\' + new_name)

    :Example:
    data_path1 = os.path.join('C:\\Users\\nicped\\Desktop\\dataset', 'grid')
    data_path = os.path.join(settings.data_path_init(), 'grid')
    audio_path = os.path.join(data_path, 'audio')
    audio_path = os.path.join(data_path, 'video')
    movie_files = file_finder(data_path1, 'wav')
    for file in movie_files:
        new_name = file.split('\\')[-2] + '_' + file.split('\\')[-1]
        copy_file_2_dir(file, audio_path + '\\' + new_name)
    """
    shutil.copy2(file, dst_dir)


def delete_attribute_from_hdf_file(file_name='lombardgrid.hdf5', group_name='test'):
    raise NotImplementedError


def delete_feature_from_hdf_file(file_name='lombardgrid.hdf5', feature='test'):
    """
    Deletes a feature or list of features from all dataset in the given hdf5 file.
    data_path = settings.actors_path_init()
    # features_to_delete = 'face_landmarks'
    features_to_delete = ('face_landmarks', 'face_landmarks_dist', 'face_landmarks_dist_mean', 'face_landmarks_motion', 'lp_env')
    delete_feature_from_hdf_file(file_name=os.path.join(data_path, 'grid.hdf5'), feature=features_to_delete)

    :param file_name:
    :param feature:
    :return:
    """
    hdf_file = h5py.File(file_name, 'r')
    temp_file_name = file_name.split('.')[0] + '_temp.' + file_name.split('.')[1]
    hdf_file_temp = h5py.File(temp_file_name, 'a')
    for group in hdf_file.keys():

        if not group in hdf_file_temp:
            hdf_file_temp.create_group(group)

        for key, value in hdf_file[group].items():
            if type(feature) == list or type(feature) == tuple:
                if key not in feature:
                    hdf_file_temp.create_dataset(group + '/' + key, data=value)
            else:
                if key != feature:
                    hdf_file_temp.create_dataset(group + '/' + key, data=value)

        for att in hdf_file[group].attrs:
            hdf_file_temp[group].attrs[att] = hdf_file[group].attrs[att]

        hdf_file_temp.flush()
    hdf_file.close()
    hdf_file_temp.close()

    # # cleanup
    os.remove(file_name)
    copy_file_2_dir(temp_file_name, file_name)
    os.remove(temp_file_name)


def delete_feature_from_hdf_file_with_if(file_name='lombardgrid.hdf5', feature='test'):
    """
    Deletes a feature or list of features from all dataset in the given hdf5 file.
    data_path = settings.actors_path_init()
    # features_to_delete = 'face_landmarks'
    features_to_delete = ('face_landmarks', 'face_landmarks_dist', 'face_landmarks_dist_mean', 'face_landmarks_motion', 'lp_env')
    delete_feature_from_hdf_file(file_name=os.path.join(data_path, 'grid.hdf5'), feature=features_to_delete)

    :param file_name:
    :param feature:
    :return:
    """
    hdf_file = h5py.File(file_name, 'r')
    temp_file_name = file_name.split('.')[0] + '_temp.' + file_name.split('.')[1]
    hdf_file_temp = h5py.File(temp_file_name, 'a')
    for group in hdf_file.keys():

        if not group in hdf_file_temp:
            hdf_file_temp.create_group(group)

        for key, value in hdf_file[group].items():
            if type(feature) == list or type(feature) == tuple:
                if key not in feature:
                    hdf_file_temp.create_dataset(group + '/' + key, data=value)
            else:
                if key == 'mod_filter1':
                    if hdf_file[group]['mod_filter1'][()].shape[-1] == 16:
                        hdf_file_temp.create_dataset(group + '/mod_filter4', data=value)
                    elif hdf_file[group]['mod_filter1'][()].shape[-1] == 25:
                        hdf_file_temp.create_dataset(group + '/mod_filter1', data=value)
                elif key == 'mod_filter2':
                    if hdf_file[group]['mod_filter2'][()].shape[-1] == 25:
                        hdf_file_temp.create_dataset(group + '/mod_filter3', data=value)
                    elif hdf_file[group]['mod_filter2'][()].shape[-1] == 16:
                        hdf_file_temp.create_dataset(group + '/mod_filter2', data=value)
                elif key == 'mod_filter_simple1':
                    if hdf_file[group]['mod_filter_simple1'][()].shape[-1] == 16:
                        hdf_file_temp.create_dataset(group + '/mod_filter_simple4', data=value)
                    elif hdf_file[group]['mod_filter_simple1'][()].shape[-1] == 25:
                        hdf_file_temp.create_dataset(group + '/mod_filter_simple1', data=value)
                elif key == 'mod_filter_simple2':
                    if hdf_file[group]['mod_filter_simple2'][()].shape[-1] == 25:
                        hdf_file_temp.create_dataset(group + '/mod_filter_simple3', data=value)
                    elif hdf_file[group]['mod_filter_simple2'][()].shape[-1] == 16:
                        hdf_file_temp.create_dataset(group + '/mod_filter_simple2', data=value)
                else:
                    hdf_file_temp.create_dataset(group + '/' + key, data=value)

        for att in hdf_file[group].attrs:
            hdf_file_temp[group].attrs[att] = hdf_file[group].attrs[att]

        hdf_file_temp.flush()
    hdf_file.close()
    hdf_file_temp.close()

    # # cleanup
    os.remove(file_name)
    copy_file_2_dir(temp_file_name, file_name)
    os.remove(temp_file_name)


def delete_feature_from_hdf_file_with_if2(file_name='lrs3_jens.hdf5', feature='mod_filter_jh1'):
    """
    Deletes a feature or list of features from all dataset in the given hdf5 file.
    data_path = settings.actors_path_init()
    # features_to_delete = 'face_landmarks'
    features_to_delete = ('face_landmarks', 'face_landmarks_dist', 'face_landmarks_dist_mean', 'face_landmarks_motion', 'lp_env')
    delete_feature_from_hdf_file(file_name=os.path.join(data_path, 'grid.hdf5'), feature=features_to_delete)

    :param file_name:
    :param feature:
    :return:
    """
    hdf_file = h5py.File(file_name, 'r')
    temp_file_name = file_name.split('.')[0] + '_temp.' + file_name.split('.')[1]
    hdf_file_temp = h5py.File(temp_file_name, 'a')
    # two_feats = ['mod_filter_jh1', 'landmarks_3d_8hz']
    two_feats = ['landmarks_3d_8hz']
    for group in hdf_file.keys():

        if not group in hdf_file_temp:
            hdf_file_temp.create_group(group)

        for key, value in hdf_file[group].items():
            if key not in two_feats:
                continue
            else:

                if len(value) != hdf_file[group].attrs['n_frames']:
                    continue
                else:
                    hdf_file_temp.create_dataset(group + '/' + key, data=value)

        for att in hdf_file[group].attrs:
            hdf_file_temp[group].attrs[att] = hdf_file[group].attrs[att]

        hdf_file_temp.flush()
    hdf_file.close()
    hdf_file_temp.close()

    # # cleanup
    os.remove(file_name)
    copy_file_2_dir(temp_file_name, file_name)
    os.remove(temp_file_name)


def change_feature_name_hdf_file(file_name='grid.hdf5', old_name='mod_filter', new_name='mod_filter_dau'):
    """
    Changes the name of a specific feature from all dataset in the given hdf5 file.
    data_path = settings.actors_path_init()
    change_feature_name_hdf_file(file_name=os.path.join(data_path, 'grid.hdf5'), old_name='mod_filter', new_name='mod_filter_dau')

    :param file_name:
    :param old_name:
    :param new_name:
    :return:
    """
    hdf_file = h5py.File(file_name, 'r')
    temp_file_name = file_name.split('.')[0] + '_temp.' + file_name.split('.')[1]
    hdf_file_temp = h5py.File(temp_file_name, 'a')
    for group in hdf_file.keys():

        if not group in hdf_file_temp:
            hdf_file_temp.create_group(group)

        for key, value in hdf_file[group].items():
            if key == old_name:
                hdf_file_temp.create_dataset(group + '/' + new_name, data=value)
            else:
                hdf_file_temp.create_dataset(group + '/' + key, data=value)

        for att in hdf_file[group].attrs:
            hdf_file_temp[group].attrs[att] = hdf_file[group].attrs[att]

        hdf_file_temp.flush()
    hdf_file.close()
    hdf_file_temp.close()

    # # cleanup
    os.remove(file_name)
    copy_file_2_dir(temp_file_name, file_name)
    os.remove(temp_file_name)


def delete_group_hdf_file(file_name='actors.hdf5', delete_name='boardcastman'):
    """
    Changes the name of a specific feature from all dataset in the given hdf5 file.
    data_path = settings.actors_path_init()
    delete_group_hdf_file(file_name=os.path.join(data_path, 'actors.hdf5'), delete_name='boardcastman')

    :param file_name:
    :param delete_name:
    :return:
    """
    hdf_file = h5py.File(file_name, 'r')
    temp_file_name = file_name.split('.')[0] + '_temp.' + file_name.split('.')[1]
    hdf_file_temp = h5py.File(temp_file_name, 'a')
    for group in hdf_file.keys():

        if group not in hdf_file_temp:
            if type(delete_name) == list or type(delete_name) == tuple:
                if group not in delete_name:
                    hdf_file_temp.create_group(group)
                else:
                    continue
            else:
                if group != delete_name:
                    hdf_file_temp.create_group(group)
                else:
                    continue

        for key, value in hdf_file[group].items():
            hdf_file_temp.create_dataset(group + '/' + key, data=value)

        for att in hdf_file[group].attrs:
            hdf_file_temp[group].attrs[att] = hdf_file[group].attrs[att]

        hdf_file_temp.flush()
    hdf_file.close()
    hdf_file_temp.close()

    # # cleanup
    os.remove(file_name)
    copy_file_2_dir(temp_file_name, file_name)
    os.remove(temp_file_name)


def handle_hdf_file(file_name, mode='delete', **kwargs):

    def delete_feature(feature):
        """
        Deletes a feature or list of features from all dataset in the given hdf5 file.
        data_path = settings.actors_path_init()
        features_to_delete = ('face_landmarks', 'face_landmarks_dist', 'face_landmarks_dist_mean', 'face_landmarks_motion', 'lp_env')
        delete_feature_from_hdf_file(file_name=os.path.join(data_path, 'grid.hdf5'), feature=features_to_delete)
        :param feature:
        :return:
        """
        if type(feature) == list or type(feature) == tuple:
            if key not in feature:
                hdf_file_temp.create_dataset(group + '/' + key, data=value)
        else:
            if key != feature:
                hdf_file_temp.create_dataset(group + '/' + key, data=value)

    def rename_feature(old, new):
        """
        Changes the name of a specific feature from all dataset in the given hdf5 file.
        data_path = settings.actors_path_init()
        fn = os.path.join(data_path, 'grid.hdf5')
        change_feature_name_hdf_file(file_name=fn, old_name='mod_filter', new_name='mod_filter_dau')
        :param old:
        :param new:
        :return:
        """
        if key == old:
            hdf_file_temp.create_dataset(group + '/' + new, data=value)
        else:
            hdf_file_temp.create_dataset(group + '/' + key, data=value)

    valid_modes = {'delete': delete_feature,
                   'rename': rename_feature}

    valid_kwargs = {'delete': ['feature'],
                    'change': ['old', 'new']}

    if mode not in valid_modes:
        raise ValueError('unknown mode, should be one of: {}'.format(valid_modes.keys()))

    for k, v in kwargs:
        if k not in valid_kwargs[mode]:
            raise ValueError('Unknown kwarg, for mode "{}", use "{}"'.format(mode, kwargs))

    temp_file_name = file_name.split('.')[0] + '_temp.' + file_name.split('.')[1]
    func = valid_modes.get(mode)
    hdf_file = h5py.File(file_name, 'r')
    hdf_file_temp = h5py.File(temp_file_name, 'a')
    for group in hdf_file.keys():
        if not group in hdf_file_temp:
            hdf_file_temp.create_group(group)
        for key, value in hdf_file[group].items():
            func(**kwargs)
        for att in hdf_file[group].attrs:
            hdf_file_temp[group].attrs[att] = hdf_file[group].attrs[att]
        hdf_file_temp.flush()

    # cleanup
    os.remove(file_name)
    copy_file_2_dir(temp_file_name, file_name)
    os.remove(temp_file_name)


def copy_features_2_other_hdf_file(features='landmarks_3d_8hz'):
    # grid_path = settings.grid_path_init()
    read_path = settings.lrs3_path_init()
    append_path = settings.lrs3_pretrain_path_init()
    hdf_file_read = h5py.File(read_path + '/lrs3.hdf5', 'r')
    hdf_file_append = h5py.File(append_path + '/lrs3_pretrain.hdf5', 'a')
    for movie in hdf_file_read.keys():
        if features not in hdf_file_append[movie].keys():
            if features in hdf_file_read[movie].keys():
                hdf_file_append.create_dataset(movie + '/' + features, data=hdf_file_read[movie][features])

            hdf_file_append[movie].attrs['n_frames'] = hdf_file_read[movie].attrs['n_frames']
            hdf_file_append[movie].attrs['fps'] = hdf_file_read[movie].attrs['fps']

        else:
            continue
    hdf_file_append.flush()
    print('All features copied')


def copy_features_2_new_hdf_file(features=['landmarks_3d_8hz', 'mod_filter']):
    read_path = settings.lrs3_path_init()
    append_path = settings.lrs3_pretrain_path_init()
    hdf_file_read = h5py.File(read_path + '/lrs3.hdf5', 'r')
    hdf_file_append = h5py.File(append_path + '/lrs3_pretrain.hdf5', 'a')

    for movie in hdf_file_read.keys():

        for feat in features:
            if feat not in hdf_file_append[movie].keys():
                if feat in hdf_file_read[movie].keys():
                    hdf_file_append.create_dataset(movie + '/' + feat, data=hdf_file_read[movie][feat][()])
                    hdf_file_append[movie].attrs['n_frames'] = hdf_file_read[movie].attrs['n_frames']
                    hdf_file_append[movie].attrs['fps'] = hdf_file_read[movie].attrs['fps']

    hdf_file_append.flush()
    print('All features copied')


if __name__ == '__main__':
    # data_path = settings.actors_path_init()
    # delete_group_hdf_file(file_name=os.path.join(data_path, 'actors.hdf5'), delete_name='m98')
    # copy_features_2_new_hdf_file(features=['landmarks_3d_8hz', 'mod_filter1'])
    # copy_features_2_other_hdf_file(features='landmarks_3d_8hz')
    # copy_features_2_new_hdf_file(features=['landmarks_3d_8hz', 'mod_filter'])
    delete_feature_from_hdf_file_with_if2(file_name='C:/Users/nicol/Documents/GitHub/PhD/data/lrs3/lrs3_jens.hdf5', feature='mod_filter_jh1')
    # audio_path = '/work3/s192241/biss/data/AV/pretrain/'
    # new_dir = '/work3/nicped/PhD/data/lrs3_trainval/'
    # make_folder(new_dir)
    # movie_files = file_finder(audio_path, 'mp4')
    # for file in movie_files:
    #     new_name = '_'.join(file.split('/')[-2:])
    #     copy_file_2_dir(file, new_dir + '/' + new_name)
