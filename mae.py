from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os
from os import listdir
import pdb
import scipy.io
import numpy as np

import tensorflow as tf
import tensorflow_compression as tfc 


def load_image(filename):
    """Loads a PNG image file."""
    string = tf.read_file(filename)
    image = tf.image.decode_image(string, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255
    return image

def quantize_image(image):
    image = tf.clip_by_value(image, 0, 1)
    image = tf.round(image * 255)
    image = tf.cast(image, tf.uint8)
    return image

def save_image(filename, image):
    """Saves an image to a PNG file."""
    image = quantize_image(image)
    string = tf.image.encode_png(image)
    return tf.write_file(filename, string)

#------------------------------------------------------------#
#-------------------  Modulation Network  -------------------#
#------------------------------------------------------------#
def modulation_network(num_filters, filters_offsets, condition, modulation_init_an=None, modulation_init_syn=None):
    with tf.variable_scope("modulation_network"):
        num_layers_a, num_layers_s = 3, 3 # change this depend on the number of layers of autoencoder
        num = np.array(num_filters)
        off = np.array(filters_offsets)
        last = num + off

        if args.condition_norm is not None:
            condition = condition / args.condition_norm  # condition normalization
        condition_tf = tf.convert_to_tensor(condition, dtype=tf.float32)
        condition_tf = tf.expand_dims(condition_tf, 0)
        condition_tf = tf.expand_dims(condition_tf, 0)

        modulation_analysis, modulation_synthesis = list(), list()

        if modulation_init_an is None:
            for i in range(num_layers_a):
                with tf.variable_scope("modulation_layer_ana%d" % i):
                    vector = scale_layer(condition_tf, last[-1])
                modulation_analysis.append(vector)
        else:
            for i in range(num_layers_a):
                with tf.variable_scope("modulation_layer_ana%d" % i):
                    vector = scale_layer(condition_tf, last[-1], modulation_init_an[i])
                modulation_analysis.append(vector)

        if modulation_init_syn is None:            
            for i in range(num_layers_s):
                with tf.variable_scope("gating_layer_syn%d" % i):
                    vector = scale_layer(condition_tf, last[-1])
                modulation_synthesis.append(vector)
        else:
            for i in range(num_layers_s):
                with tf.variable_scope("gating_layer_syn%d" % i):
                    vector = scale_layer(condition_tf, last[-1], modulation_init_syn[i])
                modulation_synthesis.append(vector)        

        return modulation_analysis, modulation_synthesis, last[-1]

def scale_layer(condition, channel, init=None, reuse=False):
    x = linear(condition, 50, scope='linear_1')
    x = tf.nn.relu(x)
    if init is None:
        x = linear(x, channel)
    else:
        x = linear(x, channel, init)
    x = tf.math.exp(x)
    return x

def linear(x, units, init=None, use_bias=True, scope='linear'):
    if args.regularizer == "L2":
        regular = tf.contrib.layers.l2_regularizer(scale=0.1)
    elif args.regularizer == "L1":
        regular = tf.contrib.layers.l1_regularizer(scale=0.1)
    else:
        regular = None
    with tf.variable_scope(scope):
        if init is None:
            x = tf.layers.dense(x, units=units, use_bias=use_bias, kernel_regularizer=regular)
        else:
            init_w = tf.constant_initializer(init)
            x = tf.layers.dense(x, units=units, kernel_initializer=init_w, use_bias=use_bias, kernel_regularizer=regular)
        return x

#------------------------------------------------------------#
#-----------------  Modulated Autoencoders  -----------------#
#------------------------------------------------------------#
def modulated_analysis_transform(tensor, conds, total_filters_num):
    """Builds the modulated analysis transform."""

    with tf.variable_scope("analysis"):
        with tf.variable_scope("layer_0"):
            layer = tfc.SignalConv2D(
                total_filters_num, (9, 9), corr=True, strides_down=4, padding="same_zeros",
                use_bias=True, activation=None)
            tensor = layer(tensor)
            vector = conds[0]
            modulated_tensor = tensor * vector

            with tf.variable_scope("gnd_an_0"):
                tensor_gdn_0 = tfc.GDN()(modulated_tensor)

        with tf.variable_scope("layer_1"):
            layer = tfc.SignalConv2D(
                total_filters_num, (5, 5), corr=True, strides_down=2, padding="same_zeros",
                use_bias=True, activation=None)
            tensor = layer(tensor_gdn_0)
            vector = conds[1]
            modulated_tensor = tensor * vector

            with tf.variable_scope("gnd_an_1"):
                tensor_gdn_1 = tfc.GDN()(modulated_tensor)

        with tf.variable_scope("layer_2"):
            layer = tfc.SignalConv2D(
                total_filters_num, (5, 5), corr=True, strides_down=2, padding="same_zeros",
                use_bias=False, activation=None)
            tensor = layer(tensor_gdn_1)
            vector = conds[2]
            modulated_tensor = tensor * vector

            with tf.variable_scope("gnd_an_2"):
                tensor_gdn_2 = tfc.GDN()(modulated_tensor)

        return tensor_gdn_2

def demodulated_synthesis_transform(tensor, conds, total_filters_num):
    """Builds the demodulated synthesis transform."""

    with tf.variable_scope("synthesis"):
        with tf.variable_scope("layer_0"):
            with tf.variable_scope("gnd_sy_0"):
                tensor_igdn_0 = tfc.GDN(inverse=True)(tensor)
            vector = conds[0]
            demodulated_tensor = tensor_igdn_0 * vector
            
            layer = tfc.SignalConv2D(
                total_filters_num, (5, 5), corr=False, strides_up=2, padding="same_zeros",
                use_bias=True, activation=None)
            tensor = layer(demodulated_tensor)

        with tf.variable_scope("layer_1"):
            with tf.variable_scope("gnd_sy_1"):
                tensor_igdn_1 = tfc.GDN(inverse=True)(tensor)
            vector = conds[1]
            demodulated_tensor = tensor_igdn_1 * vector

            layer = tfc.SignalConv2D(
                total_filters_num, (5, 5), corr=False, strides_up=2, padding="same_zeros",
                use_bias=True, activation=None)
            tensor = layer(demodulated_tensor)

        with tf.variable_scope("layer_2"):
            with tf.variable_scope("gnd_sy_2"):
                tensor_igdn_2 = tfc.GDN(inverse=True)(tensor)
            vector = conds[2]
            demodulated_tensor = tensor_igdn_2 * vector

            layer = tfc.SignalConv2D(
                3, (9, 9), corr=False, strides_up=4, padding="same_zeros",
                use_bias=True, activation=None)
            tensor = layer(demodulated_tensor)

        return tensor

#-----------------  training   -----------------#
def train():
    """Trains the model."""
    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    # Create input data pipeline.
    with tf.device('/cpu:0'):
        train_files = glob.glob(args.train_glob)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
        train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
        train_dataset = train_dataset.map(
            load_image, num_parallel_calls=args.preprocess_threads)
        train_dataset = train_dataset.map(
            lambda x: tf.random_crop(x, (args.patchsize, args.patchsize, 3)))
        train_dataset = train_dataset.batch(args.batchsize)
        train_dataset = train_dataset.prefetch(32)

    num_pixels = args.batchsize * args.patchsize ** 2

    # Get training patch from dataset.
    x = train_dataset.make_one_shot_iterator().get_next()

    # lists to keep loss for each lambda
    y, y_tilde, entropy_bottlenecks, likelihoods, x_tilde = list(), list(), list(), list(), list()
    train_bpp, train_mse, train_loss = list(), list(), list()

    # Forward pass for each RD tradeoff
    for i, _lmbda in enumerate(args.lmbda):
        with tf.variable_scope("modulation_network", reuse=(i>0)):
            cond_an, cond_syn, total_filters_num = modulation_network(args.num_filters, args.filters_offset, args.lmbda[i])

        with tf.variable_scope("analysis", reuse=(i>0)): # Reuse variables when i>0 for sharing
            _y = modulated_analysis_transform(x, cond_an, total_filters_num)
            y.append(_y)

        entropy_bottlenecks.append(tfc.EntropyBottleneck())
        _y_tilde, _likelihoods = entropy_bottlenecks[i](_y, training=True)        
        y_tilde.append(_y_tilde)
        likelihoods.append(_likelihoods)

        with tf.variable_scope("synthesis", reuse=(i > 0)):  # Reuse variable when i>0 for sharing
            _x_tilde = demodulated_synthesis_transform(y_tilde[i], cond_syn, total_filters_num)
            x_tilde.append(_x_tilde)

        # Total number of bits divided by number of pixels.
        train_bpp.append(tf.reduce_sum(tf.log(likelihoods[i])) / (-np.log(2) * num_pixels))

        # Mean squared error across pixels.
        train_mse.append(tf.reduce_mean(tf.squared_difference(x, x_tilde[i])))

        # The rate-distortion cost.
        train_loss.append(_lmbda * train_mse[i] + train_bpp[i])

    total_train_loss = tf.add_n(train_loss)

    step = tf.train.create_global_step()
    # learning_rate_placeholder_cnn = tf.placeholder(tf.float32, [], name='learning_rate_cnn')
    # learning_rate_placeholder_rate = tf.placeholder(tf.float32, [], name='learning_rate_rate')
    # main_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder_cnn)
    main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    main_step = main_optimizer.minimize(total_train_loss, global_step=step)

    aux_optimizers = list()
    list_ops = [main_step]
    for i, entropy_bottleneck in enumerate(entropy_bottlenecks):
        aux_optimizers.append(tf.train.AdamOptimizer(learning_rate=1e-3))
        # aux_optimizers.append(tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder_rate))
        list_ops.append(aux_optimizers[i].minimize(entropy_bottleneck.losses[0]))
        list_ops.append(entropy_bottleneck.updates[0])
    train_op = tf.group(list_ops)

    # Summaries
    for i, _lmbda in enumerate(args.lmbda):
        tf.summary.scalar("loss_%d" % i, train_loss[i])
        tf.summary.scalar("bpp_%d" % i, train_bpp[i])
        tf.summary.scalar("mse_%d" % i, train_mse[i]* 255 ** 2) # Rescaled
        # tf.summary.histogram("hist_layer_a0_%d" % i, features_an[i][0])
        # tf.summary.histogram("hist_layer_a1_%d" % i, features_an[i][1])
        tf.summary.histogram("hist_y_%d" % i, y[i])
        # tf.summary.image("reconstruction_%d" % i, quantize_image(x_tilde[i]))

    tf.summary.scalar("total_loss", total_train_loss)

    hooks = [
        tf.train.StopAtStepHook(last_step=args.last_step),
        tf.train.NanTensorHook(total_train_loss),
    ]

    with tf.train.MonitoredTrainingSession(
            hooks=hooks, checkpoint_dir=args.checkpoint_dir,
            save_checkpoint_secs=900, save_summaries_secs=600) as sess:
        while not sess.should_stop():
            # learning_rate_cnn = 4e-4 if step < 400000 else 2e-4
            # learning_rate_rate = 2e-3 if step < 400000 else 1e-3
            # pdb.set_trace()
            sess.run(step)
            sess.run(train_op)
            # sess.run(train_op, feed_dict={learning_rate_placeholder_cnn:learning_rate_cnn, learning_rate_placeholder_rate: learning_rate_rate})

#-----------------  evaluate   -----------------#
def evaluate():
    """Evaluate the model for test dataset"""
    # process all the images in input_path
    imagesList = listdir(args.inputPath)
    # Initialize metric scores
    bpp_actual_total, bpp_estimate_total, mse_total, psnr_total, msssim_total, msssim_db_total = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # get all the entropy models or one entropy model
    if args.model_ID is not None:
        entropy_bottlenecks = list()
        for i, _lmbda in enumerate(args.lmbda):
            entropy_bottlenecks.append(tfc.EntropyBottleneck())
        entropy_bottleneck = entropy_bottlenecks[args.model_ID]
        with tf.variable_scope("modulation_network", reuse=tf.AUTO_REUSE):
            cond_an, cond_syn, total_filters_num = modulation_network(args.num_filters, args.filters_offset, args.condition)
    else:
        print('error: model_ID is necessary for one specific entropy model')

    for image in imagesList:
        x = load_image(args.inputPath + image)
        x = tf.expand_dims(x, 0)
        x.set_shape([1, None, None, 3])

        with tf.variable_scope("analysis", reuse=tf.AUTO_REUSE): # Reuse variable when i>0 for sharing
            y = modulated_analysis_transform(x, cond_an, total_filters_num)
            
        string = entropy_bottleneck.compress(y)
        string = tf.squeeze(string, axis=0)
        y_hat, likelihoods = entropy_bottleneck(y, training=False)

        with tf.variable_scope("synthesis", reuse=tf.AUTO_REUSE): 
            x_hat_first = demodulated_synthesis_transform(y_hat, cond_syn, total_filters_num)

        num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))
        # Total number of bits divided by number of pixels.
        eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)
        # Bring both images back to 0..255 range.
        x *= 255
        x_hat = tf.clip_by_value(x_hat_first, 0, 1)
        x_hat = tf.round(x_hat * 255)
        x_hat = tf.slice(x_hat, [0, 0, 0, 0], [1,tf.shape(x)[1], tf.shape(x)[2], 3])

        mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
        psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
        msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))

        if args.save_reconstruction:
            x_shape = tf.shape(x)
            x_hat_first = x_hat_first[0, :x_shape[1], :x_shape[2], :]
            if os.path.isdir(args.outputPath):
                print(args.outputPath + ':exists.')
            else:
                os.makedirs(args.outputPath)
                print(args.outputPath + ':created.')
            op = save_image(args.outputPath + image, x_hat_first)

            with tf.Session() as sess:
                # Load the latest model checkpoint, get the evaluation results.
                latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
                tf.train.Saver().restore(sess, save_path=latest)

                string, eval_bpp, mse, psnr, msssim, num_pixels = sess.run(
                    [string, eval_bpp, mse, psnr, msssim, num_pixels])
                
                sess.run(op)

            #  The actual bits per pixel including overhead
            bpp = (8 + len(string)) * 8 / num_pixels

            print("Mean squared error: {:0.4f}".format(mse))
            print("PSNR (dB): {:0.2f}".format(psnr))
            print("Multiscale SSIM: {:0.4f}".format(msssim))
            print("Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim)))
            print("Information content in bpp: {:0.4f}".format(eval_bpp))
            print("Actual bits per pixel: {:0.4f}".format(bpp))               

            with open (args.outputPath + image[:-4] + '.txt', 'w') as f:
                f.write('Avg_bpp_actual: '+str(bpp)+'\n')
                f.write('Avg_bpp_estimate: '+str(eval_bpp)+'\n')
                f.write('Avg_mse: '+str(mse)+'\n')
                f.write('Avg_psnr: '+str(psnr)+'\n')
                f.write('Avg_msssim: '+str(msssim)+'\n')
                f.write('Avg_msssim_db: '+str(-10 * np.log10(1 - msssim))+'\n')
        else:            
            with tf.Session() as sess:
                # Load the latest model checkpoint, get the compressed string and the tensor
                # shapes.
                latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
                tf.train.Saver().restore(sess, save_path=latest)

                string, eval_bpp, mse, psnr, msssim, num_pixels = sess.run(
                    [string, eval_bpp, mse, psnr, msssim, num_pixels])

                #  The actual bits per pixel including overhead
                bpp = (8 + len(string)) * 8 / num_pixels

                print("Mean squared error: {:0.4f}".format(mse))
                print("PSNR (dB): {:0.2f}".format(psnr))
                print("Multiscale SSIM: {:0.4f}".format(msssim))
                print("Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim)))
                print("Information content in bpp: {:0.4f}".format(eval_bpp))
                print("Actual bits per pixel: {:0.4f}".format(bpp))
                
                bpp_actual_total += bpp
                bpp_estimate_total += eval_bpp
                mse_total += mse
                psnr_total += psnr
                msssim_total += msssim
                msssim_db_total += (-10 * np.log10(1 - msssim))

    if args.evaluation_name is not None:
        Avg_bpp_actual, Avg_bpp_estimate = bpp_actual_total / len(imagesList), bpp_estimate_total / len(imagesList)
        Avg_mse, Avg_psnr = mse_total / len(imagesList), psnr_total / len(imagesList)
        Avg_msssim, Avg_msssim_db = msssim_total / len(imagesList), msssim_db_total / len(imagesList)
        with open (args.evaluation_name + '.txt', 'w') as f:
            f.write('Avg_bpp_actual: '+str(Avg_bpp_actual)+'\n')
            f.write('Avg_bpp_estimate: '+str(Avg_bpp_estimate)+'\n')
            f.write('Avg_mse: '+str(Avg_mse)+'\n')
            f.write('Avg_psnr: '+str(Avg_psnr)+'\n')
            f.write('Avg_msssim: '+str(Avg_msssim)+'\n')
            f.write('Avg_msssim_db: '+str(Avg_msssim_db)+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "command", choices=["train", "evaluate"],
        help="'train' loads training data and trains (or continues "
             "to train) a new model. 'evaluate' can get the RD curves from"
             "the pretrained model.")
 
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Report bitrate and distortion when training or compressing.")
    parser.add_argument(
        "--num_filters", nargs="+", type=int, default=[128],
        help="Number of filters per layer (per R-D tradeoff point).")
    parser.add_argument(
        "--checkpoint_dir", default="train",
        help="Directory where to save/load model checkpoints.")
    parser.add_argument(
        "--train_glob", default="images/*.png",
        help="Glob pattern identifying training data. This pattern must expand "
             "to a list of RGB images in PNG format.")
    parser.add_argument(
        "--batchsize", type=int, default=8,
        help="Batch size for training.")
    parser.add_argument(
        "--patchsize", type=int, default=256,
        help="Size of image patches for training.")
    parser.add_argument(
        "--lambda", nargs="+", type=float, default=[512], dest="lmbda",
        help="Lambdas for rate-distortion tradeoff points.")
    parser.add_argument(
        "--last_step", type=int, default=1000000,
        help="Train up to this number of steps.")
    parser.add_argument(
        "--preprocess_threads", type=int, default=6,
        help="Number of CPU threads to use for parallel decoding of training "
             "images.")
    parser.add_argument(
        "--modulation_init", action="store_true",
        help="Initialize the modulation network by using the default vectors.")
    parser.add_argument(
        "--filters_offset", nargs="+", type=int, default=[0],
        help="Offset filters (per R-D tradeoff point)")
    parser.add_argument(
        "--save_reconstruction", action="store_true",
        help="save reconstructed image while evaluation")
    parser.add_argument(
        "--model_ID", type=int, default=[0],
        help="Align the model which you want to use for compression/decompression.")
    parser.add_argument(
        "--condition", type=int, default=None,
        help="condition for different RD trade-off.")
    parser.add_argument(
        "--condition_norm", type=float, default=None,
        help="Normalization of condition values.")  
    parser.add_argument(
        "--evaluation_name", type=str, default='results',
        help="the name of evaluation results txt file.")
    parser.add_argument(
        "--inputPath", type=str, default=None,
        help="Directory where to evaluation dataset.")
    parser.add_argument(
        "--outputPath", type=str, default=None,
        help="Directory where to save reconstructed images.")
    parser.add_argument(
        "--regularizer", type=str, default=None,
        help="regularizer of modulation network.")

    args = parser.parse_args()

    if args.command == "train":
        # Check consistency between lambda, num_filters, filters_offset
        if len(args.lmbda) != len(args.num_filters):
            raise ValueError("The length of lambda and num_filters should be the same.")
        if len(args.num_filters) != len(args.filters_offset):
            raise ValueError("The length num_filters and filters_offset should be the same.")
        train()
    elif args.command == "evaluate":
        if args.inputPath is None:
            raise ValueError("Need input path for evaluation.")
        evaluate()
