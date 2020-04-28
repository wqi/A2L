import argparse
import io
import keras.optimizers
import numpy as np
import os
import tensorflow as tf

from keras import backend as K
from keras.utils import multi_gpu_model
from PIL import Image
from segmentation_models import Unet

from train.data_loader import DataGenerator

'''
This script is used to train a ResNet-18-based UNet segmentation model using partially labelled
data obtained from the sampling script. Use of a distance-weighted loss can be enabled using the
--use-weights argument.

Configuration is performed via command line arguments specified below.
'''


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dir containing saved samples')
    parser.add_argument('--save-dir', type=str, required=True, default='.',
                        help='Path to dir where trained model should be saved')
    parser.add_argument('--saved-model-path', type=str,
                        help='Path to saved model from which to load weights')

    # Training Options
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=40,
                        help='Batch size used for training')
    parser.add_argument('--use-weights', action='store_true',
                        help='Specifies if saved weights should be used to compute loss/acc')

    args = parser.parse_args()

    return args


def train_model(args):
    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    # Set file paths
    args.log_path = os.path.join(args.save_dir, 'logs')
    args.model_path = os.path.join(args.save_dir, 'seg_model.h5')
    args.data_path_x = os.path.join(args.data_dir, 'x.npy')
    args.data_path_y = os.path.join(args.data_dir, 'y.npy')
    args.data_path_weights = None
    if args.use_weights:
        args.data_path_weights = os.path.join(args.data_dir, 'weights.npy')

    # Set up training data generator
    training_generator = DataGenerator(args.data_path_x, args.data_path_y, args.data_path_weights,
                                       batch_size=args.batch_size, in_shape=(256, 320, 4),
                                       out_shape=(256, 320, 1))

    # Set up model
    model = Unet('resnet18', input_shape=(256, 320, 4), activation='sigmoid', classes=1,
                 encoder_weights=None)
    adam = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                 decay=0.0)
    callbacks_list = [TensorBoardImage(args.data_path_x, args.log_path),
                      keras.callbacks.TensorBoard(log_dir=args.log_path,
                                                  update_freq=1000,
                                                  write_graph=True),
                      keras.callbacks.ModelCheckpoint(args.model_path,
                                                      save_weights_only=True)]

    # Run model on multiple GPUs if available
    try:
        model = multi_gpu_model(model)
        print("Training model on multiple GPUs")
    except ValueError:
        print("Training model on single GPU")

    # Load weights if specified
    if args.saved_model_path is not None:
        model.load_weights(args.saved_model_path)

    # Set loss
    loss = mask_loss
    acc = mask_acc
    if args.use_weights:
        loss = mask_loss_weighted
        acc = mask_acc_weighted

    # Compile model and start training
    model.compile(loss=[loss], optimizer=adam,
                  metrics=[acc])
    model.fit_generator(generator=training_generator,
                        epochs=args.epochs,
                        use_multiprocessing=True,
                        workers=8,
                        callbacks=callbacks_list)

    return model


def mask_loss(y_true, y_pred):
    weights = tf.less(y_true, 0)
    mask = tf.where(weights, tf.zeros_like(y_true), tf.ones_like(y_true))
    num_nonzero_weights = tf.cast(tf.count_nonzero(mask, axis=[1, 2, 3]) + 1, tf.float32)

    bce = K.binary_crossentropy(y_true, y_pred)
    masked_bce = tf.multiply(bce, mask)
    masked_bce = K.sum(masked_bce, axis=[1, 2, 3]) / num_nonzero_weights
    return K.mean(masked_bce)


def mask_loss_weighted(y_true, y_pred):
    mask = tf.cast(y_true[:, :, :, -1:], tf.float32) / 4
    y_true = y_true[:, :, :, :1]
    num_nonzero_weights = tf.cast(tf.count_nonzero(mask, axis=[1, 2, 3]) + 1, tf.float32)

    bce = K.binary_crossentropy(y_true, y_pred)
    masked_bce = tf.multiply(bce, mask)
    masked_bce = K.sum(masked_bce, axis=[1, 2, 3]) / num_nonzero_weights
    return K.mean(masked_bce)


def multiway_mask_loss(y_true, y_pred):
    # Mask out labels below 0 when computing loss
    weights = tf.less(K.sum(y_true, axis=3), 0)
    mask = tf.where(weights,
                    tf.zeros_like(weights, dtype=tf.float32),
                    tf.ones_like(weights, dtype=tf.float32))
    num_nonzero_weights = tf.cast(tf.count_nonzero(mask, axis=[1, 2]) + 1, tf.float32)

    # Set labels below 0 to 0 to prevent errors in cross-entropy loss computation
    invalids = tf.less(y_true, 0)
    modified_y_true = tf.where(invalids, tf.zeros_like(y_true), y_true)

    # Compute categorical cross-entropy loss
    cce = K.sparse_categorical_crossentropy(modified_y_true, y_pred)
    masked_cce = tf.multiply(cce, mask)
    masked_cce = K.sum(masked_cce, axis=[1, 2]) / num_nonzero_weights
    return K.mean(masked_cce)


def mask_acc(y_true, y_pred):
    weights = tf.less(y_true, 0)
    mask = tf.where(weights, tf.zeros_like(y_true), tf.ones_like(y_true))
    num_nonzero_weights = tf.cast(tf.count_nonzero(mask, axis=[1, 2, 3]), tf.float32)

    equal = tf.cast(K.equal(y_true, K.round(y_pred)), tf.float32)
    ksum = K.sum(equal, axis=[1, 2, 3]) / num_nonzero_weights
    return K.mean(ksum, axis=-1)


def mask_acc_weighted(y_true, y_pred):
    mask = tf.cast(y_true[:, :, :, -1:], tf.float32) / 4
    y_true = y_true[:, :, :, :1]

    equal = tf.cast(K.equal(y_true, K.round(y_pred)), tf.float32)
    weighted_equal = tf.multiply(equal, mask)
    weighted_sum = K.sum(weighted_equal, axis=[1, 2, 3]) / K.sum(mask, axis=[1, 2, 3])
    return K.mean(weighted_sum, axis=-1)


def multiway_mask_acc(y_true, y_pred):
    # Mask out labels below 0 when computing acc
    weights = tf.less(K.sum(y_true, axis=3), 0)
    mask = tf.where(weights,
                    tf.zeros_like(weights, dtype=tf.float32),
                    tf.ones_like(weights, dtype=tf.float32))
    num_nonzero_weights = tf.cast(tf.count_nonzero(mask, axis=[1, 2]) + 1, tf.float32)

    # Set labels below 0 to 0 to prevent errors in acc computation
    invalids = tf.less(y_true, 0)
    modified_y_true = tf.where(invalids, tf.zeros_like(y_true), y_true)

    # Compute accuracy
    equal = K.equal(tf.cast(K.sum(y_true, axis=3), tf.int64), tf.argmax(y_pred, axis=3))
    equal = tf.cast(equal, tf.float32)
    equal = tf.multiply(equal, mask)
    ksum = K.sum(equal, axis=[1, 2]) / num_nonzero_weights
    return K.mean(ksum, axis=-1)


class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, data_path_x, log_path):
        super().__init__()
        self.data_path_x = data_path_x
        self.log_path = log_path

    def on_epoch_end(self, epoch, logs={}):
        summary = tf.Summary(value=[tf.Summary.Value(tag='train/0',
                                    image=self._make_image(self.model, 0)),
                                    tf.Summary.Value(tag='train/10',
                                    image=self._make_image(self.model, 10)),
                                    tf.Summary.Value(tag='train/25',
                                    image=self._make_image(self.model, 25)),
                                    tf.Summary.Value(tag='train/50',
                                    image=self._make_image(self.model, 50))])
        writer = tf.summary.FileWriter(self.log_path)
        writer.add_summary(summary, epoch)
        writer.close()

    def _make_image(self, model, idx):
        height, width, channel = (256, 320, 1)
        x = np.zeros((1, 256, 320, 4))
        x[0, 8:-8, :, :] = np.load(self.data_path_x, mmap_mode='r')[idx]
        pred_y = model.predict(x)
        pred_y = np.rint(pred_y)[0, :, :, 0].astype(np.uint8)
        img = Image.fromarray(pred_y * 255, mode='L')

        output = io.BytesIO()
        img.save(output, format='PNG')
        img_string = output.getvalue()
        output.close()

        return tf.Summary.Image(height=height,
                                width=width,
                                colorspace=channel,
                                encoded_image_string=img_string)


if __name__ == "__main__":
    args = parse_arguments()
    train_model(args)
