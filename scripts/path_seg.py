#!/usr/bin/env python
# -*- coding: utf-8 -*-

# sensor_msgs::Image to numpy
#https://answers.ros.org/question/64318/how-do-i-convert-an-ros-image-into-a-numpy-array/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rospy
from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
import rospy.numpy_msg
#from .registry import converts_from_numpy, converts_to_numpy

import json
import logging
import os
import sys

import collections
import genpy

# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')

name_to_dtypes = {
    "rgb8": (np.uint8, 3),
    "rgba8": (np.uint8, 4),
    "rgb16": (np.uint16, 3),
    "rgba16": (np.uint16, 4),
    "bgr8": (np.uint8, 3),
    "bgra8": (np.uint8, 4),
    "bgr16": (np.uint16, 3),
    "bgra16": (np.uint16, 4),
    "mono8": (np.uint8, 1),
    "mono16": (np.uint16, 1),

    # for bayer image (based on cv_bridge.cpp)
    "bayer_rggb8": (np.uint8, 1),
    "bayer_bggr8": (np.uint8, 1),
    "bayer_gbrg8": (np.uint8, 1),
    "bayer_grbg8": (np.uint8, 1),
    "bayer_rggb16": (np.uint16, 1),
    "bayer_bggr16": (np.uint16, 1),
    "bayer_gbrg16": (np.uint16, 1),
    "bayer_grbg16": (np.uint16, 1),

    # OpenCV CvMat types
    "8UC1": (np.uint8, 1),
    "8UC2": (np.uint8, 2),
    "8UC3": (np.uint8, 3),
    "8UC4": (np.uint8, 4),
    "8SC1": (np.int8, 1),
    "8SC2": (np.int8, 2),
    "8SC3": (np.int8, 3),
    "8SC4": (np.int8, 4),
    "16UC1": (np.int16, 1),
    "16UC2": (np.int16, 2),
    "16UC3": (np.int16, 3),
    "16UC4": (np.int16, 4),
    "16SC1": (np.uint16, 1),
    "16SC2": (np.uint16, 2),
    "16SC3": (np.uint16, 3),
    "16SC4": (np.uint16, 4),
    "32SC1": (np.int32, 1),
    "32SC2": (np.int32, 2),
    "32SC3": (np.int32, 3),
    "32SC4": (np.int32, 4),
    "32FC1": (np.float32, 1),
    "32FC2": (np.float32, 2),
    "32FC3": (np.float32, 3),
    "32FC4": (np.float32, 4),
    "64FC1": (np.float64, 1),
    "64FC2": (np.float64, 2),
    "64FC3": (np.float64, 3),
    "64FC4": (np.float64, 4)
}

_numpy_msg = rospy.numpy_msg.numpy_msg
_cached = {}
def numpy_msg(cls):
	if cls not in _cached:
		res = _numpy_msg(cls)
		_cached[cls] = res
	else:
		res = _cached[cls]

	return res

# patch the original for good measure
rospy.numpy_msg.numpy_msg = numpy_msg

from seg_utils import seg_utils as seg

_to_numpy = {}
_from_numpy = {}

pub = None

def converts_to_numpy(msgtype, plural=False):
	assert issubclass(msgtype, genpy.Message)
	def decorator(f):
		_to_numpy[msgtype, plural] = f
		_to_numpy[numpy_msg(msgtype), plural] = f
		return f
	return decorator

def converts_from_numpy(msgtype, plural=False):
	assert issubclass(msgtype, genpy.Message)
	def decorator(f):
		_from_numpy[msgtype, plural] = f
		_from_numpy[numpy_msg(msgtype), plural] = f
		return f
	return decorator

def numpify(msg, *args, **kwargs):
	if msg is None:
		return

	conv = _to_numpy.get((msg.__class__, False))
	if not conv and isinstance(msg, collections.Sequence):
		if not msg:
			raise ValueError("Cannot determine the type of an empty Collection")
		conv = _to_numpy.get((msg[0].__class__, True))


	if not conv:
		raise ValueError("Unable to convert message {} - only supports {}".format(
			msg.__class__.__name__,
			', '.join(cls.__name__ + ("[]" if pl else '') for cls, pl in _to_numpy.keys())
		))

	return conv(msg, *args, **kwargs)

def msgify(msg_type, numpy_obj, *args, **kwargs):
	conv = _from_numpy.get((msg_type, kwargs.pop('plural', False)))
	if not conv:
		raise ValueError("Unable to build message {} - only supports {}".format(
			msg_type.__name__,
			', '.join(cls.__name__ + ("[]" if pl else '') for cls, pl in _to_numpy.keys())
		))
	return conv(numpy_obj, *args, **kwargs)

try:
    # Check whether setup was done correctly

    import tensorvision.utils as tv_utils
    import tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)

flags.DEFINE_string('logdir', None,
                    'Path to logdir.')
flags.DEFINE_string('input_image', None,
                    'Image to apply KittiSeg.')
flags.DEFINE_string('output_image', None,
                    'Image to apply KittiSeg.')

default_run = 'KittiSeg_pretrained'
weights_url = ("ftp://mi.eng.cam.ac.uk/"
               "pub/mttt2/models/KittiSeg_pretrained.zip")


@converts_to_numpy(Image)
def image_to_numpy(msg):
    if not msg.encoding in name_to_dtypes:
        raise TypeError('Unrecognized encoding {}'.format(msg.encoding))

    dtype_class, channels = name_to_dtypes[msg.encoding]
    dtype = np.dtype(dtype_class)
    dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
    shape = (msg.height, msg.width, channels)

    data = np.fromstring(msg.data, dtype=dtype).reshape(shape)
    data.strides = (
        msg.step,
        dtype.itemsize * channels,
        dtype.itemsize
    )

    if channels == 1:
        data = data[..., 0]
    return data


@converts_from_numpy(Image)
def numpy_to_image(arr, encoding):
    if not encoding in name_to_dtypes:
        raise TypeError('Unrecognized encoding {}'.format(encoding))

    im = Image(encoding=encoding)

    # extract width, height, and channels
    dtype_class, exp_channels = name_to_dtypes[encoding]
    dtype = np.dtype(dtype_class)
    if len(arr.shape) == 2:
        im.height, im.width, channels = arr.shape + (1,)
    elif len(arr.shape) == 3:
        im.height, im.width, channels = arr.shape
    else:
        raise TypeError("Array must be two or three dimensional")

    # check type and channels
    if exp_channels != channels:
        raise TypeError("Array has {} channels, {} requires {}".format(
            channels, encoding, exp_channels
        ))
    if dtype_class != arr.dtype.type:
        raise TypeError("Array is {}, {} requires {}".format(
            arr.dtype.type, encoding, dtype_class
        ))

    # make the array contiguous in memory, as mostly required by the format
    contig = np.ascontiguousarray(arr)
    im.data = contig.tostring()
    im.step = contig.strides[0]
    im.is_bigendian = (
            arr.dtype.byteorder == '>' or
            arr.dtype.byteorder == '=' and sys.byteorder == 'big'
    )

    return im


def maybe_download_and_extract(runs_dir):
    logdir = os.path.join(runs_dir, default_run)

    if os.path.exists(logdir):
        # weights are downloaded. Nothing to do
        return

    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    download_name = tv_utils.download(weights_url, runs_dir)
    logging.info("Extracting KittiSeg_pretrained.zip")

    import zipfile
    zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

    return


def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image

def callback(data):
    rospy.loginfo(rospy.get_caller_id())
    autobus = numpify(data)
    pub.publish(autobus)

def main(_):
    tv_utils.set_gpus_to_use()

    if FLAGS.logdir is None:
        # Download and use weights from the MultiNet Paper
        if 'TV_DIR_RUNS' in os.environ:
            runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                    'KittiSeg')
        else:
            runs_dir = 'RUNS'
        maybe_download_and_extract(runs_dir)
        logdir = os.path.join(runs_dir, default_run)
    else:
        logging.info("Using weights found in {}".format(FLAGS.logdir))
        logdir = FLAGS.logdir

    # Loading hyperparameters from logdir
    hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')

    logging.info("Hypes loaded successfully.")

    # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
    modules = tv_utils.load_modules_from_logdir(logdir)
    logging.info("Modules loaded successfully. Starting to build tf graph.")

    # Create tf graph and build module.
    with tf.Graph().as_default():
        # Create placeholder for input
        image_pl = tf.placeholder(tf.float32)
        image = tf.expand_dims(image_pl, 0)

        # build Tensorflow graph using the model from logdir
        prediction = core.build_inference_graph(hypes, modules,
                                                image=image)

        logging.info("Graph build successfully.")

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Load weights from logdir
        core.load_weights(logdir, sess, saver)

        logging.info("Weights loaded successfully.")

    input_image = FLAGS.input_image
    #logging.info("Starting inference using {} as input".format(input_image))
    print("Tensorflow model initialized.")

    #ros node class
    path_detect = PathDetect(hypes, sess, image_pl, prediction)
    rospy.spin()

class PathDetect:
    img_seg = None
    img_raw = None

    def __init__(self, hypes, sess, image_pl, prediction):
        self.hypes = hypes
        self.sess = sess
        self.image_pl = image_pl
        self.prediction = prediction
        rospy.init_node('path_seg', anonymous=True)
        self.pub = rospy.Publisher('/seg_img', Image, queue_size=1)
        rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)

    def callback(self, data):
        rospy.loginfo(rospy.get_caller_id())
        self.img_raw = numpify(data)
        self.run_model()
        img_seg_msg = msgify(Image, self.img_seg, encoding='rgb8')
        self.pub.publish(img_seg_msg)

    def run_model(self):
        # Load and resize input image
        image = self.img_raw
        if self.hypes['jitter']['reseize_image']:
            # Resize input only, if specified in hypes
            image_height = self.hypes['jitter']['image_height']
            image_width = self.hypes['jitter']['image_width']
            image = scp.misc.imresize(image, size=(image_height, image_width),
                                      interp='cubic')

        # Run KittiSeg model on image
        feed = {self.image_pl: image}
        softmax = self.prediction['softmax']
        output = self.sess.run([softmax], feed_dict=feed)

        # Reshape output from flat vector to 2D Image
        shape = image.shape
        output_image = output[0][:, 1].reshape(shape[0], shape[1])

        # Plot confidences as red-blue overlay
        rb_image = seg.make_overlay(image, output_image)

        # Accept all pixel with conf >= 0.5 as positive prediction
        # This creates a `hard` prediction result for class street
        threshold = 0.5
        street_prediction = output_image > threshold

        # Plot the hard prediction as green overlay
        green_image = tv_utils.fast_overlay(image, street_prediction)

        self.img_seg = green_image

if __name__ == '__main__':
    #path_seg()
    tf.app.run()