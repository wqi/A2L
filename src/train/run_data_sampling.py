import argparse
import json
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

from keras.utils import multi_gpu_model
from os.path import join
from sample import sample
from segmentation_models import Unet
from tqdm import tqdm

import common.util as util

'''
This script is used to generate self-supervised partial labels for hazard segmentation.

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
    parser.add_argument('--model-path', type=str,
                        help='Path to seed model if active sampling is being employed')
    parser.add_argument('--multi-gpu-model', action='store_true',
                        help='Specifies whether multi-GPU model is being used')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Path to dir where partially-labeled samples should be saved')

    # Sampling Options
    parser.add_argument('--samples-per-map', type=int, default=500,
                        help='Number of iterations to explore for')
    parser.add_argument('--max-sample-ratio', type=float, default=2.0,
                        help='Maximum ratio of number of attempted samples to specified \
                              samples per map')
    parser.add_argument('--freespace-only', action='store_true',
                        help='Specifies if sample goals should be selected only from freespace')

    # Visualization Options
    parser.add_argument('--viz-output', action='store_true',
                        help='Specifies whether visualizations should be shown for debugging')

    args = parser.parse_args()
    return args


def load_model(args):
    # Load trained seed segmentation model to use with active sampling
    K.set_learning_phase(0)
    model = Unet('resnet18', input_shape=(256, 320, 4), activation='sigmoid', classes=1,
                 encoder_weights=None)

    # Convert to multi-GPU model if necessary
    if args.multi_gpu_model:
        model = multi_gpu_model(model)

    model.load_weights(args.model_path)
    return model


def generate_data(args):
    '''
    Sample up to max_sample times from each map, saving results to disk in NPY format
    '''
    wad_ids = util.get_sorted_wad_ids(args.wad_dir)
    max_samples = args.samples_per_map * args.max_sample_ratio

    # Load seed model if specified
    model = None
    if args.model_path:
        model = load_model(args)

    # Sample from each map up to limit
    data_x = []
    data_y = []
    data_weights = []
    maps_sample_log = {}
    map_progress = tqdm(desc='Map Progress', total=len(wad_ids))

    for wad_id in wad_ids:
        sample_progress = tqdm(desc='Sample Progress', total=args.samples_per_map)
        valid_xy = util.get_valid_locations(args.beacon_dir, wad_id)
        game = util.setup_game(args.wad_dir, wad_id, visible=args.viz_output)

        cur_sample = 0
        times_sampled = 0
        sample_log = [0, 0, 0, 0]
        while cur_sample < args.samples_per_map and times_sampled < max_samples:
            status, rgbd_frames, seg_maps, weight_maps = sample(args, game, valid_xy, model)
            times_sampled += 1

            # Check if sample status indicates success
            if status >= 0:
                for (rgbd_frame, seg_map, weight_map) in zip(rgbd_frames, seg_maps, weight_maps):
                    # Check that labels in frame are not too sparse
                    if np.count_nonzero(seg_map) >= 1000:
                        data_x.append(rgbd_frame)
                        data_y.append(seg_map[:, :, np.newaxis])
                        data_weights.append(weight_map[:, :, np.newaxis])

                        if args.viz_output:
                            plt.imshow(rgbd_frame[:, :, :3])
                            plt.imshow(seg_map, alpha=0.6)
                            plt.show()

                sample_log[status] += 1
                cur_sample += 1
                sample_progress.update(1)

        # Log number of samples collected
        maps_sample_log[wad_id] = sample_log
        sample_progress.close()
        map_progress.update(1)
        print('INFO: Sampled map {} for {} times resulting in {} samples'.format(wad_id, times_sampled, cur_sample)) # NOQA
        print('INFO: Sample Distribution: {} success || {} environmental damage || {} monster damage || {} obstacle'.format(sample_log[0], sample_log[1], sample_log[2], sample_log[3])) # NOQA

    # Save data to disk as NPY
    data_x = np.asarray(data_x, dtype=np.uint8)
    data_y = np.asarray(data_y, dtype=np.uint8)
    data_weights = np.asarray(data_weights, dtype=np.uint8)
    np.save(join(args.save_dir, 'x.npy'), data_x)
    np.save(join(args.save_dir, 'y.npy'), data_y)
    np.save(join(args.save_dir, 'weights.npy'), data_weights)
    with open(join(args.save_dir, 'sample_stats.json'), 'w') as fp:
        json.dump(maps_sample_log, fp, indent=4)
    print('Done: Generated {} samples'.format(data_x.shape[0]))
    map_progress.close()


if __name__ == "__main__":
    args = parse_arguments()
    generate_data(args)
