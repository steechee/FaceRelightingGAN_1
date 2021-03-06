import os
import numpy as np
import tensorflow as tf
from model import BeU
import argparse
from utils import show_all_variables

flags = tf.app.flags

flags.DEFINE_integer("train_size", 45000, "The size of train images [np.inf]")
# flags.DEFINE_integer("batch_size", 1, "The size of batch images [64]")

flags.DEFINE_integer("input_h", 64, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_w", 64, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_h", 64, "The size of image to generate (will be center cropped). [108]")
flags.DEFINE_integer("output_w", 64, "The size of image to generate (will be center cropped). If None, same value as input_height [None]")

flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")

FLAGS = flags.FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    args = parser.parse_args()

    task = args.task
    log_dir = 'log/%s'%task
    summary_dir = 'summary/%s'%task
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=20, allow_soft_placement=True, log_device_placement=False))

    ae = BeU(
            sess,
            input_h = FLAGS.input_h,
            input_w = FLAGS.input_w,
            output_h = FLAGS.output_h,
            output_w = FLAGS.output_w,
            # batch_size = FLAGS.batch_size,
            log_dir = log_dir,
            sum_dir = summary_dir)

    show_all_variables()

    ae.test(istrain=False)


if __name__ == '__main__':
    main()
