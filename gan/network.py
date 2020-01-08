import tensorflow as tf
from .op import linear, batch_norm, flatten
from .output import OutputType, Normalization
import numpy as np
from enum import Enum
import os


class Network(object):
    def __init__(self, scope_name):
        self.scope_name = scope_name

    def build(self, input):
        return NotImplementedError

    @property
    def all_vars(self):
        return tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self.scope_name)

    @property
    def trainable_vars(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope_name)

    def print_layers(self):
        print("Layers of {}".format(self.scope_name))
        print(self.all_vars)

    def save(self, sess, folder):
        saver = tf.train.Saver(self.all_vars)
        path = os.path.join(folder, "model.ckpt")
        saver.save(sess, path)

    def load(self, sess, folder):
        saver = tf.train.Saver(self.all_vars)
        path = os.path.join(folder, "model.ckpt")
        saver.restore(sess, path)


class Discriminator(Network):
    def __init__(self,
                 num_layers=5, num_units=200,
                 scope_name="discriminator", *args, **kwargs):
        super(Discriminator, self).__init__(
            scope_name=scope_name, *args, **kwargs)
        self.num_layers = num_layers
        self.num_units = num_units

    def build(self, input_feature, input_attribute, train):
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            input_feature = flatten(input_feature)
            input_attribute = flatten(input_attribute)
            input_ = tf.concat([input_feature, input_attribute], 1)
            layers = [input_feature, input_attribute, input_]
            for i in range(self.num_layers - 1):
                with tf.variable_scope("layer{}".format(i)):
                    layers.append(linear(layers[-1], self.num_units))
                    layers.append(tf.nn.relu(layers[-1]))
                    # if (i > 0):
                    #    layers.append(batch_norm()(layers[-1], train=train))
            with tf.variable_scope("layer{}".format(self.num_layers - 1)):
                layers.append(linear(layers[-1], 1))
                # batch_size * 1
                layers.append(tf.squeeze(layers[-1], 1))
                # batch_size

            return layers[-1]


class AttrDiscriminator(Network):
    def __init__(self,
                 num_layers=5, num_units=200,
                 scope_name="attrDiscriminator", *args, **kwargs):
        super(AttrDiscriminator, self).__init__(
            scope_name=scope_name, *args, **kwargs)
        self.num_layers = num_layers
        self.num_units = num_units

    def build(self, input_attribute, train):
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            input_attribute = flatten(input_attribute)
            layers = [input_attribute]
            for i in range(self.num_layers - 1):
                with tf.variable_scope("layer{}".format(i)):
                    layers.append(linear(layers[-1], self.num_units))
                    layers.append(tf.nn.relu(layers[-1]))
                    # if (i > 0):
                    #    layers.append(batch_norm()(layers[-1], train=train))
            with tf.variable_scope("layer{}".format(self.num_layers - 1)):
                layers.append(linear(layers[-1], 1))
                # batch_size * 1
                layers.append(tf.squeeze(layers[-1], 1))
                # batch_size

            return layers[-1]


class RNNInitialStateType(Enum):
    ZERO = "ZERO"
    RANDOM = "RANDOM"
    VARIABLE = "VARIABLE"


class DoppelGANgerGenerator(Network):
    def __init__(self, feed_back, noise,
                 feature_outputs, attribute_outputs, real_attribute_mask,
                 sample_len,
                 attribute_num_units=100, attribute_num_layers=3,
                 feature_num_units=100, feature_num_layers=1,
                 initial_state=RNNInitialStateType.RANDOM,
                 initial_stddev=0.02,
                 scope_name="DoppelGANgerGenerator", *args, **kwargs):
        super(DoppelGANgerGenerator, self).__init__(
            scope_name=scope_name, *args, **kwargs)
        self.feed_back = feed_back
        self.noise = noise
        self.attribute_num_units = attribute_num_units
        self.attribute_num_layers = attribute_num_layers
        self.feature_num_units = feature_num_units
        self.feature_outputs = feature_outputs
        self.attribute_outputs = attribute_outputs
        self.real_attribute_mask = real_attribute_mask
        self.feature_num_layers = feature_num_layers
        self.sample_len = sample_len
        self.initial_state = initial_state
        self.initial_stddev = initial_stddev
        self.feature_out_dim = (np.sum([t.dim for t in feature_outputs]) *
                                self.sample_len)
        self.attribute_out_dim = np.sum([t.dim for t in attribute_outputs])
        if not self.noise and not self.feed_back:
            raise Exception("noise and feed_back should have at least "
                            "one True")

        self.real_attribute_outputs = []
        self.addi_attribute_outputs = []
        self.real_attribute_out_dim = 0
        self.addi_attribute_out_dim = 0
        for i in range(len(self.real_attribute_mask)):
            if self.real_attribute_mask[i]:
                self.real_attribute_outputs.append(
                    self.attribute_outputs[i])
                self.real_attribute_out_dim += self.attribute_outputs[i].dim
            else:
                self.addi_attribute_outputs.append(
                    self.attribute_outputs[i])
                self.addi_attribute_out_dim += \
                    self.attribute_outputs[i].dim

        for i in range(len(self.real_attribute_mask) - 1):
            if (self.real_attribute_mask[i] == False and
                    self.real_attribute_mask[i + 1] == True):
                raise Exception("Real attribute should come first")

        self.gen_flag_id = None
        for i in range(len(self.feature_outputs)):
            if self.feature_outputs[i].is_gen_flag:
                self.gen_flag_id = i
                break
        if self.gen_flag_id is None:
            raise Exception("cannot find gen_flag_id")
        if self.feature_outputs[self.gen_flag_id].dim != 2:
            raise Exception("gen flag output's dim should be 2")

        self.STR_REAL = "real"
        self.STR_ADDI = "addi"

    def build(self, attribute_input_noise, addi_attribute_input_noise,
              feature_input_noise, feature_input_data, train,
              attribute=None):
        with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(feature_input_noise)[0]

            if attribute is None:
                all_attribute = []
                all_discrete_attribute = []
                if len(self.addi_attribute_outputs) > 0:
                    all_attribute_input_noise = \
                        [attribute_input_noise,
                         addi_attribute_input_noise]
                    all_attribute_outputs = \
                        [self.real_attribute_outputs,
                         self.addi_attribute_outputs]
                    all_attribute_part_name = \
                        [self.STR_REAL, self.STR_ADDI]
                    all_attribute_out_dim = \
                        [self.real_attribute_out_dim,
                         self.addi_attribute_out_dim]
                else:
                    all_attribute_input_noise = [attribute_input_noise]
                    all_attribute_outputs = [self.real_attribute_outputs]
                    all_attribute_part_name = [self.STR_REAL]
                    all_attribute_out_dim = [self.real_attribute_out_dim]
            else:
                all_attribute = [attribute]
                all_discrete_attribute = [attribute]
                if len(self.addi_attribute_outputs) > 0:
                    all_attribute_input_noise = \
                        [addi_attribute_input_noise]
                    all_attribute_outputs = \
                        [self.addi_attribute_outputs]
                    all_attribute_part_name = \
                        [self.STR_ADDI]
                    all_attribute_out_dim = [self.addi_attribute_out_dim]
                else:
                    all_attribute_input_noise = []
                    all_attribute_outputs = []
                    all_attribute_part_name = []
                    all_attribute_out_dim = []

            for part_i in range(len(all_attribute_input_noise)):
                with tf.variable_scope(
                        "attribute_{}".format(all_attribute_part_name[part_i]),
                        reuse=tf.AUTO_REUSE):

                    if len(all_discrete_attribute) > 0:
                        layers = [tf.concat(
                            [all_attribute_input_noise[part_i]] +
                            all_discrete_attribute,
                            axis=1)]
                    else:
                        layers = [all_attribute_input_noise[part_i]]

                    for i in range(self.attribute_num_layers - 1):
                        with tf.variable_scope("layer{}".format(i)):
                            layers.append(
                                linear(layers[-1], self.attribute_num_units))
                            layers.append(tf.nn.relu(layers[-1]))
                            layers.append(batch_norm()(
                                layers[-1], train=train))
                    with tf.variable_scope(
                            "layer{}".format(self.attribute_num_layers - 1),
                            reuse=tf.AUTO_REUSE):
                        part_attribute = []
                        part_discrete_attribute = []
                        for i in range(len(all_attribute_outputs[part_i])):
                            with tf.variable_scope("output{}".format(i),
                                                   reuse=tf.AUTO_REUSE):
                                output = all_attribute_outputs[part_i][i]

                                sub_output_ori = linear(layers[-1], output.dim)
                                if (output.type_ == OutputType.DISCRETE):
                                    sub_output = tf.nn.softmax(sub_output_ori)
                                    sub_output_discrete = tf.one_hot(
                                        tf.argmax(sub_output, axis=1),
                                        output.dim)
                                elif (output.type_ == OutputType.CONTINUOUS):
                                    if (output.normalization ==
                                            Normalization.ZERO_ONE):
                                        sub_output = tf.nn.sigmoid(
                                            sub_output_ori)
                                    elif (output.normalization ==
                                            Normalization.MINUSONE_ONE):
                                        sub_output = tf.nn.tanh(sub_output_ori)
                                    else:
                                        raise Exception("unknown normalization"
                                                        " type")
                                    sub_output_discrete = sub_output
                                else:
                                    raise Exception("unknown output type")
                                part_attribute.append(sub_output)
                                part_discrete_attribute.append(
                                    sub_output_discrete)
                        part_attribute = tf.concat(part_attribute, axis=1)
                        part_discrete_attribute = tf.concat(
                            part_discrete_attribute, axis=1)
                        part_attribute = tf.reshape(
                            part_attribute,
                            [batch_size, all_attribute_out_dim[part_i]])
                        part_discrete_attribute = tf.reshape(
                            part_discrete_attribute,
                            [batch_size, all_attribute_out_dim[part_i]])
                        # batch_size * dim

                    part_discrete_attribute = tf.stop_gradient(
                        part_discrete_attribute)

                    all_attribute.append(part_attribute)
                    all_discrete_attribute.append(part_discrete_attribute)

            all_attribute = tf.concat(all_attribute, axis=1)
            all_discrete_attribute = tf.concat(all_discrete_attribute, axis=1)
            all_attribute = tf.reshape(
                all_attribute,
                [batch_size, self.attribute_out_dim])
            all_discrete_attribute = tf.reshape(
                all_discrete_attribute,
                [batch_size, self.attribute_out_dim])

            with tf.variable_scope("feature", reuse=tf.AUTO_REUSE):
                all_cell = []
                for i in range(self.feature_num_layers):
                    with tf.variable_scope("unit{}".format(i),
                                           reuse=tf.AUTO_REUSE):
                        cell = tf.nn.rnn_cell.LSTMCell(
                            num_units=self.feature_num_units,
                            state_is_tuple=True)
                        all_cell.append(cell)
                rnn_network = tf.nn.rnn_cell.MultiRNNCell(all_cell)

                feature_input_data_dim = \
                    len(feature_input_data.get_shape().as_list())
                if feature_input_data_dim == 3:
                    feature_input_data_reshape = tf.transpose(
                        feature_input_data, [1, 0, 2])
                feature_input_noise_reshape = tf.transpose(
                    feature_input_noise, [1, 0, 2])
                # time * batch_size * ?

                if self.initial_state == RNNInitialStateType.ZERO:
                    initial_state = rnn_network.zero_state(
                        batch_size, tf.float32)
                elif self.initial_state == RNNInitialStateType.RANDOM:
                    initial_state = tf.random_normal(
                        shape=(self.feature_num_layers,
                               2,
                               batch_size,
                               self.feature_num_units),
                        mean=0.0, stddev=1.0)
                    initial_state = tf.unstack(initial_state, axis=0)
                    initial_state = tuple(
                        [tf.nn.rnn_cell.LSTMStateTuple(
                            initial_state[idx][0], initial_state[idx][1])
                         for idx in range(self.feature_num_layers)])
                elif self.initial_state == RNNInitialStateType.VARIABLE:
                    initial_state = []
                    for i in range(self.feature_num_layers):
                        sub_initial_state1 = tf.get_variable(
                            "layer{}_initial_state1".format(i),
                            (1, self.feature_num_units),
                            initializer=tf.random_normal_initializer(
                                stddev=self.initial_stddev))
                        sub_initial_state1 = tf.tile(
                            sub_initial_state1, (batch_size, 1))
                        sub_initial_state2 = tf.get_variable(
                            "layer{}_initial_state2".format(i),
                            (1, self.feature_num_units),
                            initializer=tf.random_normal_initializer(
                                stddev=self.initial_stddev))
                        sub_initial_state2 = tf.tile(
                            sub_initial_state2, (batch_size, 1))
                        sub_initial_state = tf.nn.rnn_cell.LSTMStateTuple(
                            sub_initial_state1, sub_initial_state2)
                        initial_state.append(sub_initial_state)
                    initial_state = tuple(initial_state)
                else:
                    return NotImplementedError

                time = feature_input_noise.get_shape().as_list()[1]
                if time is None:
                    time = tf.shape(feature_input_noise)[1]

                def compute(i, state, last_output, all_output,
                            gen_flag, all_gen_flag, all_cur_argmax,
                            last_cell_output):
                    input_all = [all_discrete_attribute]
                    if self.noise:
                        input_all.append(feature_input_noise_reshape[i])
                    if self.feed_back:
                        if feature_input_data_dim == 3:
                            input_all.append(feature_input_data_reshape[i])
                        else:
                            input_all.append(last_output)
                    input_all = tf.concat(input_all, axis=1)

                    cell_new_output, new_state = rnn_network(input_all, state)
                    new_output_all = []
                    id_ = 0
                    for j in range(self.sample_len):
                        for k in range(len(self.feature_outputs)):
                            with tf.variable_scope("output{}".format(id_),
                                                   reuse=tf.AUTO_REUSE):
                                output = self.feature_outputs[k]

                                sub_output = linear(
                                    cell_new_output, output.dim)
                                if (output.type_ == OutputType.DISCRETE):
                                    sub_output = tf.nn.softmax(sub_output)
                                elif (output.type_ == OutputType.CONTINUOUS):
                                    if (output.normalization ==
                                            Normalization.ZERO_ONE):
                                        sub_output = tf.nn.sigmoid(sub_output)
                                    elif (output.normalization ==
                                            Normalization.MINUSONE_ONE):
                                        sub_output = tf.nn.tanh(sub_output)
                                    else:
                                        raise Exception("unknown normalization"
                                                        " type")
                                else:
                                    raise Exception("unknown output type")
                                new_output_all.append(sub_output)
                                id_ += 1
                    new_output = tf.concat(new_output_all, axis=1)

                    for j in range(self.sample_len):
                        all_gen_flag = all_gen_flag.write(
                            i * self.sample_len + j, gen_flag)
                        cur_gen_flag = tf.to_float(tf.equal(tf.argmax(
                            new_output_all[(j * len(self.feature_outputs) +
                                            self.gen_flag_id)],
                            axis=1), 0))
                        cur_gen_flag = tf.reshape(cur_gen_flag, [-1, 1])
                        all_cur_argmax = all_cur_argmax.write(
                            i * self.sample_len + j,
                            tf.argmax(
                                new_output_all[(j * len(self.feature_outputs) +
                                                self.gen_flag_id)],
                                axis=1))
                        gen_flag = gen_flag * cur_gen_flag

                    return (i + 1,
                            new_state,
                            new_output,
                            all_output.write(i, new_output),
                            gen_flag,
                            all_gen_flag,
                            all_cur_argmax,
                            cell_new_output)

                (i, state, _, feature, _, gen_flag, cur_argmax,
                 cell_output) = \
                    tf.while_loop(
                        lambda a, b, c, d, e, f, g, h:
                        tf.logical_and(a < time,
                                       tf.equal(tf.reduce_max(e), 1)),
                        compute,
                        (0,
                         initial_state,
                         feature_input_data if feature_input_data_dim == 2
                            else feature_input_data_reshape[0],
                         tf.TensorArray(tf.float32, time),
                         tf.ones((batch_size, 1)),
                         tf.TensorArray(tf.float32, time * self.sample_len),
                         tf.TensorArray(tf.int64, time * self.sample_len),
                         tf.zeros((batch_size, self.feature_num_units))))

                def fill_rest(i, all_output, all_gen_flag, all_cur_argmax):
                    all_output = all_output.write(
                        i, tf.zeros((batch_size, self.feature_out_dim)))

                    for j in range(self.sample_len):
                        all_gen_flag = all_gen_flag.write(
                            i * self.sample_len + j,
                            tf.zeros((batch_size, 1)))
                        all_cur_argmax = all_cur_argmax.write(
                            i * self.sample_len + j,
                            tf.zeros((batch_size,), dtype=tf.int64))
                    return (i + 1,
                            all_output,
                            all_gen_flag,
                            all_cur_argmax)

                _, feature, gen_flag, cur_argmax = tf.while_loop(
                    lambda a, b, c, d: a < time,
                    fill_rest,
                    (i, feature, gen_flag, cur_argmax))

                feature = feature.stack()
                # time * batch_size * (dim * sample_len)
                gen_flag = gen_flag.stack()
                # (time * sample_len) * batch_size * 1
                cur_argmax = cur_argmax.stack()

                gen_flag = tf.transpose(gen_flag, [1, 0, 2])
                # batch_size * (time * sample_len) * 1
                cur_argmax = tf.transpose(cur_argmax, [1, 0])
                # batch_size * (time * sample_len)
                length = tf.reduce_sum(gen_flag, [1, 2])
                # batch_size

                feature = tf.transpose(feature, [1, 0, 2])
                # batch_size * time * (dim * sample_len)
                gen_flag_t = tf.reshape(
                    gen_flag,
                    [batch_size, time, self.sample_len])
                # batch_size * time * sample_len
                gen_flag_t = tf.reduce_sum(gen_flag_t, [2])
                # batch_size * time
                gen_flag_t = tf.to_float(gen_flag_t > 0.5)
                gen_flag_t = tf.expand_dims(gen_flag_t, 2)
                # batch_size * time * 1
                gen_flag_t = tf.tile(
                    gen_flag_t,
                    [1, 1, self.feature_out_dim])
                # batch_size * time * (dim * sample_len)
                # zero out the parts after sequence ends
                feature = feature * gen_flag_t
                feature = tf.reshape(
                    feature,
                    [batch_size,
                     time * self.sample_len,
                     self.feature_out_dim / self.sample_len])
                # batch_size * (time * sample_len) * dim

            return feature, all_attribute, gen_flag, length, cur_argmax
