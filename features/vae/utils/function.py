import os
import matplotlib.pyplot as plt

from pathlib import Path

# get blockiness
def get_blockiness(chain_sequence: str):
    start_0 = list()
    start_1 = list()
    # fill start_0 and start_1 lists
    for bead_idx, element in enumerate(chain_sequence):
        if bead_idx == len(chain_sequence) - 1:  # last index
            bead_idx = -1
        if element == '0':
            start_0.append(element + chain_sequence[bead_idx + 1])
        else:
            start_1.append(element + chain_sequence[bead_idx + 1])

    # get conditional probability p_00 and p_01
    # check if len(start_0)
    if not len(start_0) == 0:
        end_0_count = 0
        for sequences in start_0:
            if sequences.endswith('0'):
                end_0_count += 1
        probability_00 = end_0_count / len(start_0)
        probability_01 = 1.0 - probability_00
    else:
        probability_00 = 0.0
        probability_01 = 0.0

    # get conditional probability p_10 and p_11
    # check if len(start_1)
    if not len(start_1) == 0:
        end_0_count = 0
        for sequences in start_1:
            if sequences.endswith('0'):
                end_0_count += 1
        probability_10 = end_0_count / len(start_1)
        probability_11 = 1.0 - probability_10
    else:
        probability_10 = 0.0
        probability_11 = 0.0

    blockiness = probability_00 * probability_11 - probability_10 * probability_01

    return blockiness


def plot_two_pairs(x_list, y_list, x_name=None, y_name=None, save_directory=None):
    plt.figure(figsize=(8, 8))
    plt.plot(x_list, y_list, 'bo')
    if x_name is not None and y_name is not None:
        plt.xlabel('%s' % x_name)
        plt.ylabel('%s' % y_name)
    if save_directory is not None:
        save_path = os.path.join(os.getcwd(), save_directory)
        if not os.path.exists(save_path):
            Path(save_path).mkdir(parents=True)
        plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_two_pairs_color_map(x_list, y_list, z_list, x_name=None, y_name=None, z_name=None, save_directory=None,
                             title=None):
    # set plot variables
    fig, ax1 = plt.subplots()
    fig.set_size_inches(8, 8)

    # labels
    if x_name is not None and y_name is not None:
        ax1.set_xlabel('%s' % x_name, fontsize=10)
        ax1.set_ylabel('%s' % y_name, fontsize=10)
    if title is not None:
        ax1.set_title('%s' % title)

    # plot
    plt.scatter(x=x_list, y=y_list, c=z_list, cmap='RdPu', alpha=1.0, s=15.0)
    color_bar = plt.colorbar()
    color_bar.set_label('%s' % z_name, fontsize=10)

    # save
    if save_directory is not None:
        save_path = os.path.join(os.getcwd(), save_directory)
        if not os.path.exists(save_path):
            Path(save_path).mkdir(parents=True)
        plt.savefig(save_path)

    plt.show()
    plt.close()
