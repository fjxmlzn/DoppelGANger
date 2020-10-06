from gpu_task_scheduler.gpu_task import GPUTask


class GANGenerateDataTask(GPUTask):
    def main(self):
        import sys
        sys.path.append("..")

        import os
        import tensorflow as tf
        from gan.load_data import load_data
        from gan.network import DoppelGANgerGenerator, Discriminator, \
            RNNInitialStateType, AttrDiscriminator
        from gan.doppelganger import DoppelGANger
        from gan import output
        from gan.util import add_gen_flag, normalize_per_sample, \
            renormalize_per_sample
        import numpy as np

        sys.modules["output"] = output

        (data_feature, data_attribute,
         data_gen_flag,
         data_feature_outputs, data_attribute_outputs) = \
            load_data(os.path.join("..", "data", self._config["dataset"]))
        print(data_feature.shape)
        print(data_attribute.shape)
        print(data_gen_flag.shape)
        num_real_attribute = len(data_attribute_outputs)

        if self._config["self_norm"]:
            (data_feature, data_attribute, data_attribute_outputs,
             real_attribute_mask) = \
                normalize_per_sample(
                    data_feature, data_attribute, data_feature_outputs,
                    data_attribute_outputs)
        else:
            real_attribute_mask = [True] * len(data_attribute_outputs)

        sample_len = self._config["sample_len"]

        data_feature, data_feature_outputs = add_gen_flag(
            data_feature,
            data_gen_flag,
            data_feature_outputs,
            sample_len)
        print(data_feature.shape)
        print(len(data_feature_outputs))

        initial_state = None
        if self._config["initial_state"] == "variable":
            initial_state = RNNInitialStateType.VARIABLE
        elif self._config["initial_state"] == "random":
            initial_state = RNNInitialStateType.RANDOM
        elif self._config["initial_state"] == "zero":
            initial_state = RNNInitialStateType.ZERO
        else:
            raise NotImplementedError
        generator = DoppelGANgerGenerator(
            feed_back=self._config["feed_back"],
            noise=self._config["noise"],
            feature_outputs=data_feature_outputs,
            attribute_outputs=data_attribute_outputs,
            real_attribute_mask=real_attribute_mask,
            sample_len=sample_len,
            feature_num_layers=self._config["gen_feature_num_layers"],
            feature_num_units=self._config["gen_feature_num_units"],
            attribute_num_layers=self._config["gen_attribute_num_layers"],
            attribute_num_units=self._config["gen_attribute_num_units"],
            initial_state=initial_state)
        discriminator = Discriminator(
            num_layers=self._config["disc_num_layers"],
            num_units=self._config["disc_num_units"])
        if self._config["aux_disc"]:
            attr_discriminator = AttrDiscriminator(
                num_layers=self._config["attr_disc_num_layers"],
                num_units=self._config["attr_disc_num_units"])

        checkpoint_dir = os.path.join(self._work_dir, "checkpoint")
        sample_dir = os.path.join(self._work_dir, "sample")
        time_path = os.path.join(self._work_dir, "time.txt")

        run_config = tf.ConfigProto()
        with tf.Session(config=run_config) as sess:
            gan = DoppelGANger(
                sess=sess,
                checkpoint_dir=checkpoint_dir,
                sample_dir=sample_dir,
                time_path=time_path,
                epoch=self._config["epoch"],
                batch_size=self._config["batch_size"],
                data_feature=data_feature,
                data_attribute=data_attribute,
                real_attribute_mask=real_attribute_mask,
                data_gen_flag=data_gen_flag,
                sample_len=sample_len,
                data_feature_outputs=data_feature_outputs,
                data_attribute_outputs=data_attribute_outputs,
                vis_freq=self._config["vis_freq"],
                vis_num_sample=self._config["vis_num_sample"],
                generator=generator,
                discriminator=discriminator,
                attr_discriminator=(attr_discriminator
                                    if self._config["aux_disc"] else None),
                d_gp_coe=self._config["d_gp_coe"],
                attr_d_gp_coe=(self._config["attr_d_gp_coe"]
                               if self._config["aux_disc"] else 0.0),
                g_attr_d_coe=(self._config["g_attr_d_coe"]
                              if self._config["aux_disc"] else 0.0),
                d_rounds=self._config["d_rounds"],
                g_rounds=self._config["g_rounds"],
                g_lr=self._config["g_lr"],
                d_lr=self._config["d_lr"],
                attr_d_lr=(self._config["attr_d_lr"]
                           if self._config["aux_disc"] else 0.0),
                extra_checkpoint_freq=self._config["extra_checkpoint_freq"],
                epoch_checkpoint_freq=self._config["epoch_checkpoint_freq"],
                num_packing=self._config["num_packing"])
            gan.build()
            print("Finished building")

            total_generate_num_sample = \
                (self._config["generate_num_train_sample"] +
                 self._config["generate_num_test_sample"])

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

            for epoch_id in range(self._config["extra_checkpoint_freq"] - 1,
                                  self._config["epoch"],
                                  self._config["extra_checkpoint_freq"]):
                print("Processing epoch_id: {}".format(epoch_id))
                mid_checkpoint_dir = os.path.join(
                    checkpoint_dir, "epoch_id-{}".format(epoch_id))
                if not os.path.exists(mid_checkpoint_dir):
                    print("Not found {}".format(mid_checkpoint_dir))
                    continue

                save_path = os.path.join(
                    self._work_dir,
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

                split = self._config["generate_num_train_sample"]

                if self._config["self_norm"]:
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
