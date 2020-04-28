import argparse
import json
import keras.backend as K
import os

from collections import defaultdict
from keras.utils import multi_gpu_model
from os.path import join
from segmentation_models import Unet

import common.util as util
from modules.navigation import navigate

'''
This script is used to evaluate navigation performance using a geometric baseline or
affordance-based approach. Experiments are specified using JSON files and use of
affordance maps is toggled via the --model-path option. Results are dumped to disk
in JSON format.

Configuration is performed via command line arguments specified below.
'''


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths
    parser.add_argument('--wad-dir', type=str, default='../data/maps/test',
                        help='Path to dir containing map files')
    parser.add_argument('--model-path', type=str,
                        help='Path to model if affordance-based navigation is employed')
    parser.add_argument('--save-dir', type=str,
                        help='Path to dir where results and vids should be saved')
    parser.add_argument('--experiment-path', type=str,
                        default='../data/experiments/navigation/demo.json',
                        help='Path to file containing experimental setup')

    # Navigation Options
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of times to repeat navigation experiment')
    parser.add_argument('--localization-noise', type=float, default=0.0)
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Max number of steps for navigation trial')

    # Config Options
    parser.add_argument('--multi-gpu-model', action='store_true',
                        help='Specifies whether multi-GPU Keras model should be used')
    parser.add_argument('--save-vid', action='store_true',
                        help='Specifies whether videos should be saved from every run')

    # Visualization Options
    parser.add_argument('--viz-output', action='store_true',
                        help='Specifies whether output should be shown for debugging')

    args = parser.parse_args()
    return args


def run_navigation_trials(args):
    # Load model if specified
    if args.model_path:
        K.set_learning_phase(1)
        model = Unet('resnet18', input_shape=(256, 320, 4), activation='sigmoid', classes=1,
                     encoder_weights=None)
        if args.multi_gpu_model:
            model = multi_gpu_model(model)

        model.load_weights(args.model_path)
    else:
        model = None

    # Create save directories for results and vids
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        if args.save_vid:
            vid_dir = join(args.save_dir, 'vids')
            os.makedirs(vid_dir, exist_ok=True)

    # Read saved experimental setup from disk
    with open(args.experiment_path, 'r') as fp:
        experiment_dict = json.load(fp)

    # Execute experiments for all maps
    result_dict = defaultdict(list)
    for wad_id, experiments in sorted(experiment_dict.items()):
        print('INFO: Testing on map {}'.format(wad_id))
        game = util.setup_game(args.wad_dir, wad_id, visible=args.viz_output,
                               localization_noise=args.localization_noise)

        # Execute individual experiments within map
        for exp_idx, experiment in enumerate(experiments):
            for i in range(args.iterations):
                util.setup_trial(game, experiment['start'])
                vid_path = join(vid_dir, '{}_{}_{}.mp4'.format(wad_id, exp_idx, i)) if args.save_vid else None # NOQA
                result, full_path = navigate(game, args.max_steps, experiment['goal'], model, vid_path)
                result_dict[wad_id].append(result)
                print('INFO: Trial complete {}'.format(result))

        # Save results from experiment
        if args.save_dir:
            result_path = join(args.save_dir, 'results.json')
            with open(result_path, 'w') as fp:
                json.dump(result_dict, fp, sort_keys=True, indent=4)


if __name__ == "__main__":
    args = parse_arguments()
    run_navigation_trials(args)
