import argparse
import copy
import numpy as np
import os

from os.path import join

from run_data_sampling import generate_data
from run_train_model import train_model

'''
This script is used to train a ResNet-18-based UNet segmentation model using active learning.
A small dataset is first collected and used to train a seed model, which is employed to actively
generate additional examples that maximize model uncertainty. The next iterations of this model
are then trained using a blended dataset consisting of exanokes collected in all active iterations.

Configuration is performed via command line arguments specified below.
'''


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths
    parser.add_argument('--wad-dir', type=str, default='../data/maps/test',
                        help='Path to dir containing map files')
    parser.add_argument('--beacon-dir', type=str, required=True,
                        help='Path to dir containing beacons specifying valid spawn positions \
                              for each map')
    parser.add_argument('--multi-gpu-model', action='store_true',
                        help='Specifies whether multi-GPU model should be used')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Path to root dir where collected samples and intermediate models \
                              should be saved')

    # Sampling Options
    parser.add_argument('--active-iterations', type=int, default=5,
                        help='Number of active learning iterations to run')
    parser.add_argument('--samples-per-map', type=int, default=500,
                        help='Total number of samples to collect per map')
    parser.add_argument('--max-sample-ratio', type=float, default=2.0,
                        help='Ratio of maximum number of times to sample a map')
    parser.add_argument('--freespace-only', action='store_true',
                        help='Specifies if sample goals should be selected only from freespace')

    # Training Options
    parser.add_argument('--batch-size', type=int, default=40,
                        help='Batch size used for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train for in first iteration')

    args = parser.parse_args()
    return args


def active_train_loop(args):
    '''
    Run active sampling + training loop active_iterations times.
    '''
    for idx in range(args.active_iterations):
        active_iteration(args, idx)


def active_iteration(args, idx):
    '''
    Collect samples using seed model and traing next iteration of active learning model.
    '''
    # Create dir for first iteration samples/model
    cur_sample_dir = join(args.save_dir, str(idx), 'samples')
    os.makedirs(cur_sample_dir, exist_ok=True)

    # Collect samples from environments (using previously trained model if not first iteration)
    sample_args = copy.deepcopy(args)
    sample_args.samples_per_map = int(args.samples_per_map / args.active_iterations)
    sample_args.save_dir = cur_sample_dir
    sample_args.viz_output = False
    if idx > 0:
        prev_model_path = join(args.save_dir, str(idx - 1), 'seg_model.h5')
        sample_args.model_path = prev_model_path
    else:
        sample_args.model_path = None

    print('INFO: Collecting samples for iteration {}'.format(idx))
    generate_data(sample_args)
    print('INFO: Done collecting samples for iteration {}'.format(idx))

    # Build blended dataset from current and previous iterations
    if idx > 0:
        print('INFO: Building blended dataset for iteration {}'.format(idx))
        build_blended_dataset(args, idx)
        print('INFO: Done building blended dataset for iteration {}'.format(idx))

    # Train model using saved data from current iteration
    # (init weights from previous model if not first iteration)
    train_args = copy.deepcopy(args)
    train_args.save_dir = join(args.save_dir, str(idx))
    train_args.use_weights = True
    if idx > 0:
        prev_model_path = join(args.save_dir, str(idx - 1), 'seg_model.h5')
        blended_data_path = join(args.save_dir, str(idx), 'blended_samples')
        train_args.saved_model_path = prev_model_path
        train_args.data_dir = blended_data_path
    else:
        train_args.saved_model_path = None
        train_args.data_dir = cur_sample_dir

    print('INFO: Training model for iteration {}'.format(idx))
    train_model(train_args)
    print('INFO: Done training model for iteration {}'.format(idx))


def build_blended_dataset(args, idx):
    '''
    Blend samples collected in current iteration of active loop with previously-collected samples.
    '''
    blended_dir = join(args.save_dir, str(idx), 'blended_samples')
    os.makedirs(blended_dir, exist_ok=True)

    # Load memmaps for current samples
    cur_sample_dir = join(args.save_dir, str(idx), 'samples')
    cur_x = np.load(join(cur_sample_dir, 'x.npy'), mmap_mode='r')
    cur_y = np.load(join(cur_sample_dir, 'y.npy'), mmap_mode='r')
    cur_weights = np.load(join(cur_sample_dir, 'weights.npy'), mmap_mode='r')

    # Load memmaps for previous samples
    prev_sample_dir = join(args.save_dir, str(idx - 1), 'samples')
    prev_x = np.load(join(prev_sample_dir, 'x.npy'), mmap_mode='r')
    prev_y = np.load(join(prev_sample_dir, 'y.npy'), mmap_mode='r')
    prev_weights = np.load(join(prev_sample_dir, 'weights.npy'), mmap_mode='r')

    # Build blended dataset
    num_new_samples = cur_x.shape[0]
    num_old_samples = min(prev_x.shape[0], num_new_samples)
    rand_old_idxs = np.random.choice(prev_x.shape[0], size=num_old_samples, replace=False)

    # Blend X data
    rand_old_x = prev_x[rand_old_idxs, :, :, :]
    blended_x = np.concatenate((cur_x, rand_old_x), axis=0)
    np.save(join(blended_dir, 'x.npy'), blended_x)
    del rand_old_x
    del blended_x

    # Blend y data
    rand_old_y = prev_y[rand_old_idxs, :, :, :]
    blended_y = np.concatenate((cur_y, rand_old_y), axis=0)
    np.save(join(blended_dir, 'y.npy'), blended_y)
    del rand_old_y
    del blended_y

    # Blend weight data
    rand_old_weights = prev_weights[rand_old_idxs, :, :, :]
    blended_weights = np.concatenate((cur_weights, rand_old_weights), axis=0)
    np.save(join(blended_dir, 'weights.npy'), blended_weights)
    del rand_old_weights
    del blended_weights


if __name__ == "__main__":
    args = parse_arguments()
    active_train_loop(args)
