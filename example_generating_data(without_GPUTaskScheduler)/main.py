import sys
sys.path.append("..")

from gan import output
sys.modules["output"] = output

from gan.doppelganger import DoppelGANger
from gan.util import add_gen_flag, normalize_per_sample, renormalize_per_sample
from gan.load_data import load_data
from gan.network import DoppelGANgerGenerator, Discriminator, AttrDiscriminator
import os
import tensorflow as tf
import numpy as np


if __name__ == "__main__":
    sample_len = 10

    (data_feature, data_attribute,
     data_gen_flag,
     data_feature_outputs, data_attribute_outputs) = \
        load_data(os.path.join("..", "data", "web"))
    print(data_feature.shape)
    print(data_attribute.shape)
    print(data_gen_flag.shape)
    num_real_attribute = len(data_attribute_outputs)

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

    checkpoint_dir = os.path.join("..", "test", "checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    sample_dir = os.path.join("..", "test", "sample")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    time_path = os.path.join("..", "test", "time.txt")
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

    generate_num_train_sample = 50000
    generate_num_test_sample = 50000

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
        print("Finished building")

        total_generate_num_sample = \
            (generate_num_train_sample +
             generate_num_test_sample)

        if data_feature.shape[1] % sample_len != 0:
            raise Exception("length must be a multiple of sample_len")
        length = int(data_feature.shape[1] / sample_len)
        real_attribute_input_noise = gan.gen_attribute_input_noise(
            total_generate_num_sample)
        addi_attribute_input_noise = gan.gen_attribute_input_noise(
            total_generate_num_sample)
        feature_input_noise = gan.gen_feature_input_noise(
            total_generate_num_sample, length)
        input_data = gan.gen_feature_input_data_free(
            total_generate_num_sample)

        for epoch_id in range(extra_checkpoint_freq - 1,
                              epoch,
                              extra_checkpoint_freq):
            print("Processing epoch_id: {}".format(epoch_id))
            mid_checkpoint_dir = os.path.join(
                checkpoint_dir, "epoch_id-{}".format(epoch_id))
            if not os.path.exists(mid_checkpoint_dir):
                print("Not found {}".format(mid_checkpoint_dir))
                continue

            save_path = os.path.join(
                "..",
                "test",
                "generated_samples",
                "epoch_id-{}".format(epoch_id))
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            train_path_ori = os.path.join(
                save_path, "generated_data_train_ori.npz")
            test_path_ori = os.path.join(
                save_path, "generated_data_test_ori.npz")
            train_path = os.path.join(
                save_path, "generated_data_train.npz")
            test_path = os.path.join(
                save_path, "generated_data_test.npz")
            if os.path.exists(test_path):
                print("Save_path {} exists".format(save_path))
                continue

            gan.load(mid_checkpoint_dir)

            print("Finished loading")

            features, attributes, gen_flags, lengths = gan.sample_from(
                real_attribute_input_noise, addi_attribute_input_noise,
                feature_input_noise, input_data)
            # specify given_attribute parameter, if you want to generate
            # data according to an attribute
            print(features.shape)
            print(attributes.shape)
            print(gen_flags.shape)
            print(lengths.shape)

            split = generate_num_train_sample

            np.savez(
                train_path_ori,
                data_feature=features[0: split],
                data_attribute=attributes[0: split],
                data_gen_flag=gen_flags[0: split])
            np.savez(
                test_path_ori,
                data_feature=features[split:],
                data_attribute=attributes[split:],
                data_gen_flag=gen_flags[split:])

            features, attributes = renormalize_per_sample(
                features, attributes, data_feature_outputs,
                data_attribute_outputs, gen_flags,
                num_real_attribute=num_real_attribute)
            print(features.shape)
            print(attributes.shape)

            np.savez(
                train_path,
                data_feature=features[0: split],
                data_attribute=attributes[0: split],
                data_gen_flag=gen_flags[0: split])
            np.savez(
                test_path,
                data_feature=features[split:],
                data_attribute=attributes[split:],
                data_gen_flag=gen_flags[split:])

            print("Done")
