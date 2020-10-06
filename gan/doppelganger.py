import tensorflow as tf
import numpy as np
from tqdm import tqdm
import datetime
import os
import math
from .util import draw_feature, draw_attribute

class DoppelGANger(object):
    def __init__(self, sess, checkpoint_dir, sample_dir, time_path,
                 epoch, batch_size,
                 data_feature, data_attribute, real_attribute_mask,
                 data_gen_flag,
                 sample_len, data_feature_outputs, data_attribute_outputs,
                 vis_freq, vis_num_sample,
                 generator, discriminator,
                 d_rounds, g_rounds, d_gp_coe,
                 extra_checkpoint_freq, num_packing,
                 attr_discriminator=None,
                 attr_d_gp_coe=None, g_attr_d_coe=None,
                 epoch_checkpoint_freq=1,
                 attribute_latent_dim=5, feature_latent_dim=5,
                 fix_feature_network=False,
                 g_lr=0.001, g_beta1=0.5,
                 d_lr=0.001, d_beta1=0.5,
                 attr_d_lr=0.001, attr_d_beta1=0.5):
        """Constructor of DoppelGANger
        Args:
            sess: A tensorflow session
            checkpoint_dir: Directory to save model checkpoints and logs
            sample_dir: Directory to save the visualizations of generated
                samples during training
            time_path: File path for saving epoch timestamps
            epoch: Number of training epochs
            batch_size: Training batch size
            data_feature: Training features, in numpy float32 array format.
                The size is [(number of training samples) x (maximum length) x
                (total dimension of features)]. The last two dimensions of 
                features are for indicating whether the time series has already 
                ended. [1, 0] means the time series does not end at this time
                step (i.e., the time series is still activated at the next time
                step). [0, 1] means the time series ends exactly at this time 
                step or has ended before. The features are padded by zeros 
                after the last activated batch.
                For example, 
                (1) assume maximum length is 6, and sample_len (the time series
                batch size) is 3:
                (1.1) If the length of a sample is 1, the last two dimensions
                of features should be: 
                [[0, 1],[0, 1],[0, 1],[0, 0],[0, 0],[0, 0]]
                (1.2) If the length of a sample is 3, the last two dimensions
                of features should be: 
                [[1, 0],[1, 0],[0, 1],[0, 0],[0, 0],[0, 0]]
                (1.3) If the length of a sample is 4, the last two dimensions
                of features should be:
                [[1, 0],[1, 0],[1, 0],[0, 1],[0, 1],[0, 1]]
                (2) assume maximum length is 6, and sample_len (the time series
                batch size) is 1:
                (1.1) If the length of a sample is 1, the last two dimensions
                of features should be: 
                [[0, 1],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]]
                (1.2) If the length of a sample is 3, the last two dimensions
                of features should be: 
                [[1, 0],[1, 0],[0, 1],[0, 0],[0, 0],[0, 0]]
                (1.3) If the length of a sample is 4, the last two dimensions
                of features should be:
                [[1, 0],[1, 0],[1, 0],[0, 1],[0, 0],[0, 0]]
                Actually, you do not need to deal with generating those two
                dimensions. Function util.add_gen_flag does the job of adding
                those two dimensions to the original data.
                Those two dimensions are for enabling DoppelGANger to generate
                samples with different length
            data_attribute: Training attributes, in numpy float32 array format.
                The size is [(number of training samples) x (total dimension 
                of attributes)]
            real_attribute_mask: List of True/False, the length equals the 
                number of attributes. False if the attribute is (max-min)/2 or
                (max+min)/2, True otherwise
            data_gen_flag: Flags indicating the activation of features, in 
                numpy float32 array format. The size is [(number of training 
                samples) x (maximum length)]. 1 means the time series is 
                activated at this time step, 0 means the time series is 
                inactivated at this timestep. 
                For example, 
                (1) assume maximum length is 6:
                (1.1) If the length of a sample is 1, the flags should be: 
                [1, 0, 0, 0, 0, 0]
                (1.2) If the length of a sample is 3, the flags should be:
                [1, 1, 1, 0, 0, 0]
                Different from the last two dimensions of data_feature, the
                values of data_gen_flag does not influenced by sample_len
            sample_len: The time series batch size
            data_feature_outputs: A list of Output objects, indicating the 
                dimension, type, normalization of each feature
            data_attribute_outputs A list of Output objects, indicating the 
                dimension, type, normalization of each attribute
            vis_freq: The frequency of visualizing generated samples during 
                training (unit: training batch)
            vis_num_sample: The number of samples to visualize each time during
                training
            generator: An instance of network.DoppelGANgerGenerator
            discriminator: An instance of network.Discriminator
            d_rounds: Number of discriminator steps per batch
            g_rounds: Number of generator steps per batch
            d_gp_coe: Weight of gradient penalty loss in Wasserstein GAN
            extra_checkpoint_freq: The frequency of saving the trained model in
                a separated folder (unit: epoch)
            num_packing: Packing degree in PacGAN (a method for solving mode
                collapse in NeurIPS 2018, see https://arxiv.org/abs/1712.04086)
            attr_discriminator: An instance of network.AttrDiscriminator. None
                if you do not want to use this auxiliary discriminator
            attr_d_gp_coe: Weight of gradient penalty loss in Wasserstein GAN
                for the auxiliary discriminator
            g_attr_d_coe: Weight of the auxiliary discriminator in the
                generator's loss
            epoch_checkpoint_freq: The frequency of saving the trained model 
                (unit: epoch)
            attribute_latent_dim: The dimension of noise for generating 
                attributes
            feature_latent_dim: The dimension of noise for generating 
                features
            fix_feature_network: Whether to fix the feature network during 
                training
            g_lr: The learning rate in Adam for training the generator
            g_beta1: The beta1 in Adam for training the generator 
            d_lr: The learning rate in Adam for training the discriminator
            d_beta1: The beta1 in Adam for training the discriminator 
            attr_d_lr: The learning rate in Adam for training the auxiliary
                discriminator
            attr_d_beta1: The beta1 in Adam for training the auxiliary
                discriminator 
        """
        self.sess = sess
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.time_path = time_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.data_feature = data_feature
        self.data_attribute = data_attribute
        self.real_attribute_mask = real_attribute_mask
        self.data_gen_flag = data_gen_flag
        self.sample_len = sample_len
        self.data_feature_outputs = data_feature_outputs
        self.data_attribute_outputs = data_attribute_outputs
        self.vis_freq = vis_freq
        self.vis_num_sample = vis_num_sample
        self.generator = generator
        self.discriminator = discriminator
        self.attr_discriminator = attr_discriminator
        self.d_rounds = d_rounds
        self.g_rounds = g_rounds
        self.d_gp_coe = d_gp_coe
        self.attr_d_gp_coe = attr_d_gp_coe
        self.g_attr_d_coe = g_attr_d_coe
        self.extra_checkpoint_freq = extra_checkpoint_freq
        self.num_packing = num_packing
        self.epoch_checkpoint_freq = epoch_checkpoint_freq
        self.attribute_latent_dim = attribute_latent_dim
        self.feature_latent_dim = feature_latent_dim
        self.fix_feature_network = fix_feature_network
        self.g_lr = g_lr
        self.g_beta1 = g_beta1
        self.d_lr = d_lr
        self.d_beta1 = d_beta1
        self.attr_d_lr = attr_d_lr
        self.attr_d_beta1 = attr_d_beta1

        self.check_data()

        if self.data_feature.shape[1] % self.sample_len != 0:
            raise Exception("length must be a multiple of sample_len")
        self.sample_time = int(self.data_feature.shape[1] / self.sample_len)
        self.sample_feature_dim = self.data_feature.shape[2]
        self.sample_attribute_dim = self.data_attribute.shape[1]
        self.sample_real_attribute_dim = 0
        for i in range(len(self.real_attribute_mask)):
            if self.real_attribute_mask[i]:
                self.sample_real_attribute_dim += \
                    self.data_attribute_outputs[i].dim

        self.EPS = 1e-8

        self.MODEL_NAME = "model"

    def check_data(self):
        self.gen_flag_dims = []

        dim = 0
        for output in self.data_feature_outputs:
            if output.is_gen_flag:
                if output.dim != 2:
                    raise Exception("gen flag output's dim should be 2")
                self.gen_flag_dims = [dim, dim + 1]
                break
            dim += output.dim
        if len(self.gen_flag_dims) == 0:
            raise Exception("gen flag not found")

        if (self.data_feature.shape[2] !=
                np.sum([t.dim for t in self.data_feature_outputs])):
            raise Exception(
                "feature dimension does not match data_feature_outputs")

        if len(self.data_gen_flag.shape) != 2:
            raise Exception("data_gen_flag should be 2 dimension")

        self.data_gen_flag = np.expand_dims(self.data_gen_flag, 2)

    def build(self):
        self.build_connection()
        self.build_loss()
        self.build_summary()
        self.saver = tf.train.Saver()

    def build_connection(self):
        # build connections for train-fake
        self.g_feature_input_noise_train_pl_l = []
        for i in range(self.num_packing):
            self.g_feature_input_noise_train_pl_l.append(
                tf.placeholder(
                    tf.float32,
                    [None, self.sample_time, self.feature_latent_dim],
                    name="g_feature_input_noise_train_{}".format(i)))
        self.g_real_attribute_input_noise_train_pl_l = []
        for i in range(self.num_packing):
            self.g_real_attribute_input_noise_train_pl_l.append(
                tf.placeholder(
                    tf.float32,
                    [None, self.attribute_latent_dim],
                    name="g_real_attribute_input_noise_train_{}".format(i)))
        self.g_addi_attribute_input_noise_train_pl_l = []
        for i in range(self.num_packing):
            self.g_addi_attribute_input_noise_train_pl_l.append(
                tf.placeholder(
                    tf.float32,
                    [None, self.attribute_latent_dim],
                    name=("g_addi_attribute_input_noise_train_{}".format(i))))
        self.g_feature_input_data_train_pl_l = []
        for i in range(self.num_packing):
            self.g_feature_input_data_train_pl_l.append(
                tf.placeholder(
                    tf.float32,
                    [None, self.sample_len * self.sample_feature_dim],
                    name="g_feature_input_data_train_{}".format(i)))

        batch_size = tf.shape(self.g_feature_input_noise_train_pl_l[0])[0]

        self.real_attribute_mask_tensor = []
        for i in range(len(self.real_attribute_mask)):
            if self.real_attribute_mask[i]:
                sub_mask_tensor = tf.ones(
                    (batch_size, self.data_attribute_outputs[i].dim))
            else:
                sub_mask_tensor = tf.zeros(
                    (batch_size, self.data_attribute_outputs[i].dim))
            self.real_attribute_mask_tensor.append(sub_mask_tensor)
        self.real_attribute_mask_tensor = tf.concat(
            self.real_attribute_mask_tensor,
            axis=1)

        self.g_output_feature_train_tf_l = []
        self.g_output_attribute_train_tf_l = []
        self.g_output_gen_flag_train_tf_l = []
        self.g_output_length_train_tf_l = []
        self.g_output_argmax_train_tf_l = []
        for i in range(self.num_packing):
            (g_output_feature_train_tf, g_output_attribute_train_tf,
             g_output_gen_flag_train_tf, g_output_length_train_tf,
             g_output_argmax_train_tf) = \
                self.generator.build(
                    self.g_real_attribute_input_noise_train_pl_l[i],
                    self.g_addi_attribute_input_noise_train_pl_l[i],
                    self.g_feature_input_noise_train_pl_l[i],
                    self.g_feature_input_data_train_pl_l[i],
                    train=True)
            # g_output_feature_train_tf: batch_size * (time * sample_len) * dim
            # g_output_attribute_train_tf: batch_size * dim
            # g_output_gen_flag_train_tf: batch_size * (time * sample_len) * 1
            # g_output_length_train_tf: batch_size
            if self.fix_feature_network:
                g_output_feature_train_tf = tf.zeros_like(
                    g_output_feature_train_tf)
                g_output_gen_flag_train_tf = tf.zeros_like(
                    g_output_gen_flag_train_tf)
                g_output_attribute_train_tf *= self.real_attribute_mask_tensor

            self.g_output_feature_train_tf_l.append(
                g_output_feature_train_tf)
            self.g_output_attribute_train_tf_l.append(
                g_output_attribute_train_tf)
            self.g_output_gen_flag_train_tf_l.append(
                g_output_gen_flag_train_tf)
            self.g_output_length_train_tf_l.append(
                g_output_length_train_tf)
            self.g_output_argmax_train_tf_l.append(
                g_output_argmax_train_tf)
        self.g_output_feature_train_tf = tf.concat(
            self.g_output_feature_train_tf_l,
            axis=1)
        self.g_output_attribute_train_tf = tf.concat(
            self.g_output_attribute_train_tf_l,
            axis=1)

        self.d_fake_train_tf = self.discriminator.build(
            self.g_output_feature_train_tf,
            self.g_output_attribute_train_tf,
            train=True)

        if self.attr_discriminator is not None:
            self.attr_d_fake_train_tf = self.attr_discriminator.build(
                self.g_output_attribute_train_tf,
                train=True)

        self.real_feature_pl_l = []
        for i in range(self.num_packing):
            real_feature_pl = tf.placeholder(
                tf.float32,
                [None,
                 self.sample_time * self.sample_len,
                 self.sample_feature_dim],
                name="real_feature_{}".format(i))
            if self.fix_feature_network:
                real_feature_pl = tf.zeros_like(
                    real_feature_pl)
            self.real_feature_pl_l.append(real_feature_pl)
        self.real_attribute_pl_l = []
        for i in range(self.num_packing):
            real_attribute_pl = tf.placeholder(
                tf.float32,
                [None, self.sample_attribute_dim],
                name="real_attribute_{}".format(i))
            if self.fix_feature_network:
                real_attribute_pl *= self.real_attribute_mask_tensor
            self.real_attribute_pl_l.append(real_attribute_pl)
        self.real_feature_pl = tf.concat(
            self.real_feature_pl_l,
            axis=1)
        self.real_attribute_pl = tf.concat(
            self.real_attribute_pl_l,
            axis=1)

        self.d_real_train_tf = self.discriminator.build(
            self.real_feature_pl,
            self.real_attribute_pl,
            train=True)
        self.d_real_test_tf = self.discriminator.build(
            self.real_feature_pl,
            self.real_attribute_pl,
            train=False)

        if self.attr_discriminator is not None:
            self.attr_d_real_train_tf = self.attr_discriminator.build(
                self.real_attribute_pl,
                train=True)

        self.g_real_attribute_input_noise_test_pl = tf.placeholder(
            tf.float32,
            [None, self.attribute_latent_dim],
            name="g_real_attribute_input_noise_test")
        self.g_addi_attribute_input_noise_test_pl = tf.placeholder(
            tf.float32,
            [None, self.attribute_latent_dim],
            name="g_addi_attribute_input_noise_test")
        self.g_feature_input_noise_test_pl = tf.placeholder(
            tf.float32,
            [None, None, self.feature_latent_dim],
            name="g_feature_input_noise_test")

        self.g_feature_input_data_test_teacher_pl = tf.placeholder(
            tf.float32,
            [None, None, self.sample_len * self.sample_feature_dim],
            name="g_feature_input_data_test_teacher")
        (self.g_output_feature_test_teacher_tf,
         self.g_output_attribute_test_teacher_tf,
         self.g_output_gen_flag_test_teacher_tf,
         self.g_output_length_test_teacher_tf, _) = \
            self.generator.build(
                self.g_real_attribute_input_noise_test_pl,
                self.g_addi_attribute_input_noise_test_pl,
                self.g_feature_input_noise_test_pl,
                self.g_feature_input_data_test_teacher_pl,
                train=False)

        self.g_feature_input_data_test_free_pl = tf.placeholder(
            tf.float32,
            [None, self.sample_len * self.sample_feature_dim],
            name="g_feature_input_data_test_free")
        (self.g_output_feature_test_free_tf,
         self.g_output_attribute_test_free_tf,
         self.g_output_gen_flag_test_free_tf,
         self.g_output_length_test_free_tf, _) = \
            self.generator.build(
                self.g_real_attribute_input_noise_test_pl,
                self.g_addi_attribute_input_noise_test_pl,
                self.g_feature_input_noise_test_pl,
                self.g_feature_input_data_test_free_pl,
                train=False)

        self.given_attribute_attribute_pl = tf.placeholder(
            tf.float32,
            [None, self.sample_real_attribute_dim],
            name="given_attribute")
        (self.g_output_feature_given_attribute_test_free_tf,
         self.g_output_attribute_given_attribute_test_free_tf,
         self.g_output_gen_flag_given_attribute_test_free_tf,
         self.g_output_length_given_attribute_test_free_tf, _) = \
            self.generator.build(
                None,
                self.g_addi_attribute_input_noise_test_pl,
                self.g_feature_input_noise_test_pl,
                self.g_feature_input_data_test_free_pl,
                train=False,
                attribute=self.given_attribute_attribute_pl)

        self.generator.print_layers()
        self.discriminator.print_layers()
        if self.attr_discriminator is not None:
            self.attr_discriminator.print_layers()

    def build_loss(self):
        batch_size = tf.shape(self.g_feature_input_noise_train_pl_l[0])[0]

        self.g_loss_d = -tf.reduce_mean(self.d_fake_train_tf)
        if self.attr_discriminator is not None:
            self.g_loss_attr_d = -tf.reduce_mean(self.attr_d_fake_train_tf)
            self.g_loss = (self.g_loss_d +
                           self.g_attr_d_coe * self.g_loss_attr_d)
        else:
            self.g_loss = self.g_loss_d

        self.d_loss_fake = tf.reduce_mean(self.d_fake_train_tf)
        self.d_loss_real = -tf.reduce_mean(self.d_real_train_tf)
        alpha_dim2 = tf.random_uniform(
            shape=[batch_size, 1],
            minval=0.,
            maxval=1.)
        alpha_dim3 = tf.expand_dims(alpha_dim2, 2)
        differences_input_feature = (self.g_output_feature_train_tf -
                                     self.real_feature_pl)
        interpolates_input_feature = (self.real_feature_pl +
                                      alpha_dim3 * differences_input_feature)
        differences_input_attribute = (self.g_output_attribute_train_tf -
                                       self.real_attribute_pl)
        interpolates_input_attribute = (self.real_attribute_pl +
                                        (alpha_dim2 *
                                         differences_input_attribute))
        gradients = tf.gradients(
            self.discriminator.build(
                interpolates_input_feature,
                interpolates_input_attribute,
                train=True),
            [interpolates_input_feature, interpolates_input_attribute])
        slopes1 = tf.reduce_sum(tf.square(gradients[0]),
                                reduction_indices=[1, 2])
        slopes2 = tf.reduce_sum(tf.square(gradients[1]),
                                reduction_indices=[1])
        slopes = tf.sqrt(slopes1 + slopes2 + self.EPS)
        self.d_loss_gp = tf.reduce_mean((slopes - 1.)**2)

        self.d_loss = (self.d_loss_fake +
                       self.d_loss_real +
                       self.d_gp_coe * self.d_loss_gp)

        if self.attr_discriminator is not None:
            self.attr_d_loss_fake = tf.reduce_mean(self.attr_d_fake_train_tf)
            self.attr_d_loss_real = -tf.reduce_mean(self.attr_d_real_train_tf)
            alpha_dim2 = tf.random_uniform(
                shape=[batch_size, 1],
                minval=0.,
                maxval=1.)
            differences_input_attribute = (self.g_output_attribute_train_tf -
                                           self.real_attribute_pl)
            interpolates_input_attribute = (self.real_attribute_pl +
                                            (alpha_dim2 *
                                             differences_input_attribute))
            gradients = tf.gradients(
                self.attr_discriminator.build(
                    interpolates_input_attribute,
                    train=True),
                [interpolates_input_attribute])
            slopes1 = tf.reduce_sum(tf.square(gradients[0]),
                                    reduction_indices=[1])
            slopes = tf.sqrt(slopes1 + self.EPS)
            self.attr_d_loss_gp = tf.reduce_mean((slopes - 1.)**2)

            self.attr_d_loss = (self.attr_d_loss_fake +
                                self.attr_d_loss_real +
                                self.attr_d_gp_coe * self.attr_d_loss_gp)

        self.g_op = \
            tf.train.AdamOptimizer(self.g_lr, self.g_beta1)\
            .minimize(
                self.g_loss,
                var_list=self.generator.trainable_vars)

        self.d_op = \
            tf.train.AdamOptimizer(self.d_lr, self.d_beta1)\
            .minimize(
                self.d_loss,
                var_list=self.discriminator.trainable_vars)

        if self.attr_discriminator is not None:
            self.attr_d_op = \
                tf.train.AdamOptimizer(self.attr_d_lr, self.attr_d_beta1)\
                .minimize(
                    self.attr_d_loss,
                    var_list=self.attr_discriminator.trainable_vars)

    def build_summary(self):
        self.g_summary = []
        self.g_summary.append(tf.summary.scalar(
            "loss/g/d", self.g_loss_d))
        if self.attr_discriminator is not None:
            self.g_summary.append(tf.summary.scalar(
                "loss/g/attr_d", self.g_loss_attr_d))
        self.g_summary.append(tf.summary.scalar(
            "loss/g", self.g_loss))
        self.g_summary = tf.summary.merge(self.g_summary)

        self.d_summary = []
        self.d_summary.append(tf.summary.scalar(
            "loss/d/fake", self.d_loss_fake))
        self.d_summary.append(tf.summary.scalar(
            "loss/d/real", self.d_loss_real))
        self.d_summary.append(tf.summary.scalar(
            "loss/d/gp", self.d_loss_gp))
        self.d_summary.append(tf.summary.scalar(
            "loss/d", self.d_loss))
        self.d_summary.append(tf.summary.scalar(
            "d/fake", tf.reduce_mean(self.d_fake_train_tf)))
        self.d_summary.append(tf.summary.scalar(
            "d/real", tf.reduce_mean(self.d_real_train_tf)))
        self.d_summary = tf.summary.merge(self.d_summary)

        if self.attr_discriminator is not None:
            self.attr_d_summary = []
            self.attr_d_summary.append(tf.summary.scalar(
                "loss/attr_d/fake", self.attr_d_loss_fake))
            self.attr_d_summary.append(tf.summary.scalar(
                "loss/attr_d/real", self.attr_d_loss_real))
            self.attr_d_summary.append(tf.summary.scalar(
                "loss/attr_d/gp", self.attr_d_loss_gp))
            self.attr_d_summary.append(tf.summary.scalar(
                "loss/attr_d", self.attr_d_loss))
            self.attr_d_summary.append(tf.summary.scalar(
                "attr_d/fake", tf.reduce_mean(self.attr_d_fake_train_tf)))
            self.attr_d_summary.append(tf.summary.scalar(
                "attr_d/real", tf.reduce_mean(self.attr_d_real_train_tf)))
            self.attr_d_summary = tf.summary.merge(self.attr_d_summary)

    def save(self, global_id, saver=None, checkpoint_dir=None):
        if saver is None:
            saver = self.saver
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        saver.save(
            self.sess,
            os.path.join(checkpoint_dir, self.MODEL_NAME),
            global_step=global_id)

    def load(self, checkpoint_dir=None):
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        # In cases where people move the checkpoint directory to another place,
        # model path indicated by get_checkpoint_state will be wrong. So we
        # get the model name and then recontruct path using checkpoint_dir
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        global_id = int(ckpt_name[len(self.MODEL_NAME) + 1:])
        return global_id

    def discriminate_from(self, real_features, real_attributes):
        results = []
        round_ = int(
            math.ceil(float(real_features[0].shape[0]) / self.batch_size))
        for i in range(round_):
            feed_dict = {}
            for j in range(self.num_packing):
                batch_data_feature = real_features[j][
                    i * self.batch_size:
                    (i + 1) * self.batch_size]
                batch_data_attribute = real_attributes[j][
                    i * self.batch_size:
                    (i + 1) * self.batch_size]

                feed_dict[self.real_feature_pl_l[j]] = \
                    batch_data_feature
                feed_dict[self.real_attribute_pl_l[j]] = \
                    batch_data_attribute
            sub_results = self.sess.run(
                self.d_real_test_tf, feed_dict=feed_dict)
            results.append(sub_results)

        results = np.concatenate(results, axis=0)
        return results

    def sample_from(self, real_attribute_input_noise,
                    addi_attribute_input_noise, feature_input_noise,
                    feature_input_data, given_attribute=None,
                    return_gen_flag_feature=False):
        features = []
        attributes = []
        gen_flags = []
        lengths = []
        round_ = int(
            math.ceil(float(feature_input_noise.shape[0]) / self.batch_size))
        for i in range(round_):
            if given_attribute is None:
                if feature_input_data.ndim == 2:
                    (sub_features, sub_attributes, sub_gen_flags,
                     sub_lengths) = self.sess.run(
                        [self.g_output_feature_test_free_tf,
                         self.g_output_attribute_test_free_tf,
                         self.g_output_gen_flag_test_free_tf,
                         self.g_output_length_test_free_tf],
                        feed_dict={
                            self.g_real_attribute_input_noise_test_pl:
                                real_attribute_input_noise[
                                    i * self.batch_size:
                                    (i + 1) * self.batch_size],
                            self.g_addi_attribute_input_noise_test_pl:
                                addi_attribute_input_noise[
                                    i * self.batch_size:
                                    (i + 1) * self.batch_size],
                            self.g_feature_input_noise_test_pl:
                                feature_input_noise[
                                    i * self.batch_size:
                                    (i + 1) * self.batch_size],
                            self.g_feature_input_data_test_free_pl:
                                feature_input_data[
                                    i * self.batch_size:
                                    (i + 1) * self.batch_size]})
                else:
                    (sub_features, sub_attributes, sub_gen_flags,
                     sub_lengths) = self.sess.run(
                        [self.g_output_feature_test_teacher_tf,
                         self.g_output_attribute_test_teacher_tf,
                         self.g_output_gen_flag_test_teacher_tf,
                         self.g_output_length_test_teacher_tf],
                        feed_dict={
                            self.g_real_attribute_input_noise_test_pl:
                                real_attribute_input_noise[
                                    i * self.batch_size:
                                    (i + 1) * self.batch_size],
                            self.g_addi_attribute_input_noise_test_pl:
                                addi_attribute_input_noise[
                                    i * self.batch_size:
                                    (i + 1) * self.batch_size],
                            self.g_feature_input_noise_test_pl:
                                feature_input_noise[
                                    i * self.batch_size:
                                    (i + 1) * self.batch_size],
                            self.g_feature_input_data_test_teacher_pl:
                                feature_input_data[
                                    i * self.batch_size:
                                    (i + 1) * self.batch_size]})
            else:
                (sub_features, sub_attributes, sub_gen_flags,
                 sub_lengths) = self.sess.run(
                    [self.g_output_feature_given_attribute_test_free_tf,
                     self.g_output_attribute_given_attribute_test_free_tf,
                     self.g_output_gen_flag_given_attribute_test_free_tf,
                     self.g_output_length_given_attribute_test_free_tf],
                    feed_dict={
                        self.g_addi_attribute_input_noise_test_pl:
                            addi_attribute_input_noise[
                                i * self.batch_size:
                                (i + 1) * self.batch_size],
                        self.g_feature_input_noise_test_pl:
                            feature_input_noise[
                                i * self.batch_size:
                                (i + 1) * self.batch_size],
                        self.g_feature_input_data_test_free_pl:
                            feature_input_data[
                                i * self.batch_size:
                                (i + 1) * self.batch_size],
                        self.given_attribute_attribute_pl:
                            given_attribute[
                                i * self.batch_size:
                                (i + 1) * self.batch_size]})
            features.append(sub_features)
            attributes.append(sub_attributes)
            gen_flags.append(sub_gen_flags)
            lengths.append(sub_lengths)

        features = np.concatenate(features, axis=0)
        attributes = np.concatenate(attributes, axis=0)
        gen_flags = np.concatenate(gen_flags, axis=0)
        lengths = np.concatenate(lengths, axis=0)

        if not return_gen_flag_feature:
            features = np.delete(features, self.gen_flag_dims, axis=2)

        assert len(gen_flags.shape) == 3
        assert gen_flags.shape[2] == 1
        gen_flags = gen_flags[:, :, 0]

        return features, attributes, gen_flags, lengths

    def gen_attribute_input_noise(self, num_sample):
        return np.random.normal(
            size=[num_sample, self.attribute_latent_dim])

    def gen_feature_input_noise(self, num_sample, length):
        return np.random.normal(
            size=[num_sample, length, self.feature_latent_dim])

    def gen_feature_input_data_free(self, num_sample):
        return np.zeros(
            [num_sample, self.sample_len * self.sample_feature_dim],
            dtype=np.float32)

    def gen_feature_input_data_teacher(self, num_sample):
        id_ = np.random.choice(
            self.data_feature.shape[0], num_sample, replace=False)
        data_feature_ori = self.data_feature[id_, :, :]
        data_feature = np.reshape(
            data_feature_ori,
            [num_sample,
             self.sample_time,
             self.sample_len * self.sample_feature_dim])
        input_ = np.concatenate(
            [np.zeros(
                [num_sample, 1, self.sample_len * self.sample_feature_dim],
                dtype=np.float32),
             data_feature[:, :-1, :]],
            axis=1)
        ground_truth_feature = data_feature_ori
        ground_truth_length = np.sum(self.data_gen_flag[id_, :, :],
                                     axis=(1, 2))
        ground_truth_attribute = self.data_attribute[id_, :]
        return (input_, ground_truth_feature, ground_truth_attribute,
                ground_truth_length)

    def visualize(self, epoch_id, batch_id, global_id):
        def sub1(features, attributes, lengths,
                 ground_truth_features, ground_truth_attributes,
                 ground_truth_lengths, type_):
            file_path = os.path.join(
                self.sample_dir,
                "epoch_id-{},batch_id-{},global_id-{},type-{},samples.npz"
                .format(epoch_id, batch_id, global_id, type_))
            np.savez(file_path,
                     features=features, attributes=attributes, lengths=lengths,
                     ground_truth_features=ground_truth_features,
                     ground_truth_attributes=ground_truth_attributes,
                     ground_truth_lengths=ground_truth_lengths)

            file_path = os.path.join(
                self.sample_dir,
                "epoch_id-{},batch_id-{},global_id-{},type-{},feature"
                .format(epoch_id, batch_id, global_id, type_))
            if ground_truth_features is None:
                draw_feature(
                    features,
                    lengths,
                    self.data_feature_outputs,
                    file_path)
            else:
                draw_feature(
                    np.concatenate([features, ground_truth_features], axis=0),
                    np.concatenate([lengths, ground_truth_lengths], axis=0),
                    self.data_feature_outputs,
                    file_path)

            file_path = os.path.join(
                self.sample_dir,
                "epoch_id-{},batch_id-{},global_id-{},type-{},attribute"
                .format(epoch_id, batch_id, global_id, type_))
            if ground_truth_features is None:
                draw_attribute(
                    attributes,
                    self.data_attribute_outputs,
                    file_path)
            else:
                draw_attribute(
                    np.concatenate([attributes, ground_truth_attributes],
                                   axis=0),
                    self.data_attribute_outputs,
                    file_path)

        real_attribute_input_noise = self.gen_attribute_input_noise(
            self.vis_num_sample)
        addi_attribute_input_noise = self.gen_attribute_input_noise(
            self.vis_num_sample)
        feature_input_noise = self.gen_feature_input_noise(
            self.vis_num_sample, self.sample_time)

        feature_input_data = self.gen_feature_input_data_free(
            self.vis_num_sample)
        features, attributes, gen_flags, lengths = self.sample_from(
            real_attribute_input_noise, addi_attribute_input_noise,
            feature_input_noise, feature_input_data,
            return_gen_flag_feature=True)
        # print(list(features[0]))
        # print(list(gen_flags[0]))
        # print(lengths[0])
        # exit()
        sub1(features, attributes, lengths, None, None, None, "free")

        (feature_input_data, ground_truth_feature, ground_truth_attribute,
         ground_truth_length) = \
            self.gen_feature_input_data_teacher(self.vis_num_sample)
        features, attributes, gen_flags, lengths = self.sample_from(
            real_attribute_input_noise, addi_attribute_input_noise,
            feature_input_noise, feature_input_data,
            return_gen_flag_feature=True)
        sub1(features, attributes, lengths,
             ground_truth_feature, ground_truth_attribute, ground_truth_length,
             "teacher")

    def train(self, feature_network_checkpoint_path=None, restore=False):
        tf.global_variables_initializer().run()
        if restore is True:
            restore_global_id = self.load()
            print("Loaded from global_id {}".format(restore_global_id))
        else:
            restore_global_id = -1

        if feature_network_checkpoint_path is not None:
            # feature
            variables = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=self.generator.scope_name + "/feature")
            print(variables)
            saver = tf.train.Saver(variables)
            saver.restore(self.sess, feature_network_checkpoint_path)

            # min max
            variables = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=self.generator.scope_name + "/attribute_addi")
            print(variables)
            if len(variables) > 0:
                saver = tf.train.Saver(variables)
                saver.restore(self.sess, feature_network_checkpoint_path)

        self.summary_writer = tf.summary.FileWriter(
            self.checkpoint_dir, self.sess.graph)

        batch_num = self.data_feature.shape[0] // self.batch_size

        global_id = 0

        for epoch_id in tqdm(range(self.epoch)):
            data_id = np.random.choice(
                self.data_feature.shape[0],
                size=(self.data_feature.shape[0], self.num_packing))

            if global_id > restore_global_id:
                if ((epoch_id + 1) % self.epoch_checkpoint_freq == 0 or
                        epoch_id == self.epoch - 1):
                    with open(self.time_path, "a") as f:
                        time = datetime.datetime.now().strftime(
                            '%Y-%m-%d %H:%M:%S.%f')
                        f.write("epoch {} starts: {}\n".format(epoch_id, time))

            for batch_id in range(batch_num):
                feed_dict = {}
                for i in range(self.num_packing):
                    batch_data_id = data_id[batch_id * self.batch_size:
                                            (batch_id + 1) * self.batch_size,
                                            i]
                    batch_data_feature = self.data_feature[batch_data_id]
                    batch_data_attribute = self.data_attribute[batch_data_id]

                    batch_real_attribute_input_noise = \
                        self.gen_attribute_input_noise(self.batch_size)
                    batch_addi_attribute_input_noise = \
                        self.gen_attribute_input_noise(self.batch_size)
                    batch_feature_input_noise = \
                        self.gen_feature_input_noise(
                            self.batch_size, self.sample_time)
                    batch_feature_input_data = \
                        self.gen_feature_input_data_free(self.batch_size)

                    feed_dict[self.real_feature_pl_l[i]] = \
                        batch_data_feature
                    feed_dict[self.real_attribute_pl_l[i]] = \
                        batch_data_attribute
                    feed_dict[self.
                              g_real_attribute_input_noise_train_pl_l[i]] = \
                        batch_real_attribute_input_noise
                    feed_dict[self.
                              g_addi_attribute_input_noise_train_pl_l[i]] = \
                        batch_addi_attribute_input_noise
                    feed_dict[self.g_feature_input_noise_train_pl_l[i]] = \
                        batch_feature_input_noise
                    feed_dict[self.g_feature_input_data_train_pl_l[i]] = \
                        batch_feature_input_data

                if global_id > restore_global_id:
                    for _ in range(self.d_rounds - 1):
                        self.sess.run(self.d_op, feed_dict=feed_dict)
                        if self.attr_discriminator is not None:
                            self.sess.run(self.attr_d_op, feed_dict=feed_dict)
                    summary_result, _ = self.sess.run(
                        [self.d_summary, self.d_op],
                        feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_result, global_id)
                    if self.attr_discriminator is not None:
                        summary_result, _ = self.sess.run(
                            [self.attr_d_summary, self.attr_d_op],
                            feed_dict=feed_dict)
                        self.summary_writer.add_summary(
                            summary_result, global_id)

                    for _ in range(self.g_rounds - 1):
                        self.sess.run(self.g_op, feed_dict=feed_dict)
                    summary_result, _ = self.sess.run(
                        [self.g_summary, self.g_op],
                        feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_result, global_id)

                    if (batch_id + 1) % self.vis_freq == 0:
                        self.visualize(epoch_id, batch_id, global_id)

                global_id += 1

            if global_id - 1 > restore_global_id:
                if ((epoch_id + 1) % self.epoch_checkpoint_freq == 0 or
                        epoch_id == self.epoch - 1):
                    self.visualize(epoch_id, -1, global_id - 1)
                    self.save(global_id - 1)
                    with open(self.time_path, "a") as f:
                        time = datetime.datetime.now().strftime(
                            '%Y-%m-%d %H:%M:%S.%f')
                        f.write("epoch {} ends: {}\n".format(epoch_id, time))

                if (epoch_id + 1) % self.extra_checkpoint_freq == 0:
                    saver = tf.train.Saver()
                    checkpoint_dir = os.path.join(
                        self.checkpoint_dir,
                        "epoch_id-{}".format(epoch_id))
                    self.save(global_id - 1, saver, checkpoint_dir)
