import sys
sys.path.append("..")

from gan import output
sys.modules["output"] = output

from gan.doppelganger import DoppelGANger
from gan.util import add_gen_flag, normalize_per_sample
from gan.load_data import load_data
from gan.network import DoppelGANgerGenerator, Discriminator, AttrDiscriminator
import os
import tensorflow as tf


if __name__ == "__main__":
    sample_len = 10

    (data_feature, data_attribute,
     data_gen_flag,
     data_feature_outputs, data_attribute_outputs) = \
        load_data("../data/web")
    print(data_feature.shape)
    print(data_attribute.shape)
    print(data_gen_flag.shape)

    (data_feature, data_attribute, data_attribute_outputs,
     real_attribute_mask) = \
        normalize_per_sample(
            data_feature, data_attribute, data_feature_outputs,
            data_attribute_outputs)
    print(real_attribute_mask)
    print(data_feature.shape)
    print(data_attribute.shape)
    print(len(data_attribute_outputs))

    data_feature, data_feature_outputs = add_gen_flag(
        data_feature, data_gen_flag, data_feature_outputs, sample_len)
    print(data_feature.shape)
    print(len(data_feature_outputs))

    generator = DoppelGANgerGenerator(
        feed_back=False,
        noise=True,
        feature_outputs=data_feature_outputs,
        attribute_outputs=data_attribute_outputs,
        real_attribute_mask=real_attribute_mask,
        sample_len=sample_len)
    discriminator = Discriminator()
    attr_discriminator = AttrDiscriminator()

    checkpoint_dir = "./test/checkpoint"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    sample_dir = "./test/sample"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    time_path = "./test/time.txt"
    epoch = 400
    batch_size = 100
    vis_freq = 200
    vis_num_sample = 5
    d_rounds = 1
    g_rounds = 1
    d_gp_coe = 10.0
    attr_d_gp_coe = 10.0
    g_attr_d_coe = 1.0
    extra_checkpoint_freq = 5
    num_packing = 1

    run_config = tf.ConfigProto()
    with tf.Session(config=run_config) as sess:
        gan = DoppelGANger(
            sess=sess,
            checkpoint_dir=checkpoint_dir,
            sample_dir=sample_dir,
            time_path=time_path,
            epoch=epoch,
            batch_size=batch_size,
            data_feature=data_feature,
            data_attribute=data_attribute,
            real_attribute_mask=real_attribute_mask,
            data_gen_flag=data_gen_flag,
            sample_len=sample_len,
            data_feature_outputs=data_feature_outputs,
            data_attribute_outputs=data_attribute_outputs,
            vis_freq=vis_freq,
            vis_num_sample=vis_num_sample,
            generator=generator,
            discriminator=discriminator,
            attr_discriminator=attr_discriminator,
            d_gp_coe=d_gp_coe,
            attr_d_gp_coe=attr_d_gp_coe,
            g_attr_d_coe=g_attr_d_coe,
            d_rounds=d_rounds,
            g_rounds=g_rounds,
            num_packing=num_packing,
            extra_checkpoint_freq=extra_checkpoint_freq)
        gan.build()
        gan.train()
