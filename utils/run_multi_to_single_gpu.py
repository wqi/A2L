import argparse
import keras.backend as K

from keras.utils import multi_gpu_model
from segmentation_models import Unet

'''
This script is used to convert segmentation models trained on multiple GPUs to a format compatible
with single GPU machines.

Configuration is performed via command line arguments specified below.
'''


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to multi-GPU model')
    parser.add_argument('--out-path', type=str,
                        help='Output path for converted single-GPU model')

    args = parser.parse_args()
    return args


def convert_model(args):
    if args.out_path is None:
        args.out_path = args.model_path.split('.h5')[0] + '_single.h5'

    # Load multi-GPU model weights
    K.set_learning_phase(1)
    model = Unet('resnet18', input_shape=(256, 320, 4), activation='sigmoid', classes=1,
                 encoder_weights=None)
    model = multi_gpu_model(model)
    model.load_weights(args.model_path)

    # Set weights in single-GPU model and save
    single_model = model.layers[-2]
    single_model.save(args.out_path)


if __name__ == "__main__":
    args = parse_arguments()
    convert_model(args)
