from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# import tensorflow as tf
# from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cv2

from python_code.utils.file_utils import increment_folder_number, make_folder
# from python_code.utils.face_detection import load_normalized_face_landmarks, DlibFaceUtils
from python_code import settings

# Using seaborn's style
plt.style.use('seaborn-whitegrid')
# plt.style.use('seaborn-white')
# plt.style.use('tex')

# Old fonts
# tex_fonts = {
#     # Use LaTeX to write all text
#     "text.usetex": True,
#     "font.family": "serif",
#     # Use 10pt font in plots, to match 10pt font in document
#     "axes.labelsize": 8,
#     "font.size": 8,
#     # Make the legend/label fonts a little smaller
#     "legend.fontsize": 7,
#     "xtick.labelsize": 7,
#     "ytick.labelsize": 7,
#     "axes.titlesize": 8,
#     "figure.titlesize": 8
# }

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": False,
    "font.family": "arial",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 9,
    "font.size": 9,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.titlesize": 9,
    "figure.titlesize": 9
}

plt.rcParams.update(tex_fonts)


def plot_or_save(save_fig=False, save_path='/', save_name='average'):

    if save_fig:
        make_folder(save_path)
        fname = save_name + '.png'
        # plt.tight_layout()
        # plt.tight_layout(h_pad=0.1)
        plt.savefig(os.path.normpath(os.path.join(save_path, fname)), bbox_inches='tight')
        print('Saved', fname)
        plt.close()
    else:
        plt.show()
        plt.close()


def set_fig_size(width='article', fraction=1, subplots=(1, 1), adjusted=False, adjusted2=False):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'article':
        width_pt = 430.00462
    elif width == 'report':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    if width == 'column':
        fig_width_in = 5.2
    elif width == 'full':
        fig_width_in = 7.5
    else:
        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt

    if adjusted:
        # Figure height in inches when wanting to plot freq and landmarks together
        fig_height_in = fig_width_in * (golden_ratio + golden_ratio*0.5) * (subplots[0] / subplots[1])
    elif adjusted2:
        # Figure height in inches when wanting to plot freq, landmarks and XYZ together
        fig_height_in = fig_width_in * (golden_ratio + golden_ratio*1) * (subplots[0] / subplots[1])
        if fig_height_in > 8.75:
            fig_height_in = 8.75
            fig_width_in = fig_height_in / ((golden_ratio + golden_ratio*1) * (subplots[0] / subplots[1]))
    else:
        # Figure height in inches when wanting golden ratio
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def set_subfig_size(comp, adjusted=False, adjusted2=False, fraction=1):
    # Define the size of the subplot
    # if 8 >= comp > 4:
    #     rows = 2
    # elif 12 >= comp > 8:
    #     rows = 3
    # elif comp > 12:
    #     rows = 4
    # else:
    #     rows = 1
    # columns = int(np.ceil(comp / rows))

    if comp > 4:
        columns3 = comp % 3
        columns4 = comp % 4

        # if the two are equally low, it will prefer four columns
        all_columns = [columns4, columns3]
        if 0 == np.argmin(all_columns):
            columns = 4
        elif 1 == np.argmin(all_columns):
            columns = 3
    else:
        columns = comp
    rows = int(np.ceil(comp / columns))
    return rows, columns, set_fig_size(fraction=fraction, subplots=(rows, columns), adjusted=adjusted, adjusted2=adjusted2)


# def make_video_from_landmarks(audio_cca, visual_cca, title='title'):
#
#     landmarks = load_normalized_face_landmarks()
#     landmarks -= np.mean(landmarks, axis=0)
#
#     max_landmarks = np.max(landmarks, axis=0)
#     min_landmarks = np.min(landmarks, axis=0)
#
#     max_landmarks += 0.1
#     min_landmarks -= 0.1
#     landmarks[:, 1] = -landmarks[:, 1]
#
#     ells = [Ellipse(xy=landmarks[i, :], width=0.02, height=0.06, angle=0) for i in range(len(landmarks))]
#
#     # ells = [Ellipse(xy=landmarks[i, :],
#     #                 width=0.02, height=0.06,
#     #                 angle=np.random.rand() * 360)
#     #         for i in range(len(landmarks))]
#
#     # fig = plt.figure()
#     fig, ax1 = plt.subplots(subplot_kw={'aspect': 'equal'})
#     for e in ells:
#         ax1.add_artist(e)
#         e.set_clip_box(ax1.bbox)
#         e.set_alpha(1)  # how transparent or pastel the color should be
#         e.set_facecolor(np.random.rand(3))
#
#     ax1.set_xlim(min_landmarks[0], max_landmarks[0])
#     ax1.set_ylim(-max_landmarks[1], -min_landmarks[0])
#     # plt.axes(xlim=(min_landmarks[0], max_landmarks[0]), ylim=(-max_landmarks[1], -min_landmarks[0]))
#     # plt.scatter(landmarks[:, 0], -landmarks[:, 1])
#     plt.show()
#     # fname = '_tmp%03d.png' % ii
#     # print('Saving frame', fname)
#     # plt.savefig(fname)
#     # fig.clf()


# def plot_landmarks_on_image():
#     data_path = settings.grid_path_init() + '/plots/old_plots/face.png'
#     write_path = settings.grid_path_init() + '/plots/old_plots/face_landmarks1.png'
#     image = cv2.imread(data_path)
#     face_utils = DlibFaceUtils()
#     face_utils(image)
#     face_landmarks = face_utils._get_landmarks()
#     color = (0, 0, 255)
#     for xy_corr in face_landmarks[0]:
#         cv2.circle(image, tuple(xy_corr), 6, color, -1)
#     # cv2.imshow('Video', image)
#     # cv2.waitKey(0)
#     cv2.imwrite(write_path, image)


if __name__ == '__main__':
    plot_landmarks_on_image()
    # make_video_from_landmarks()
