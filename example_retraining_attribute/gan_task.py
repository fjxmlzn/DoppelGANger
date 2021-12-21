from gpu_task_scheduler.gpu_task import GPUTask


class GANTask(GPUTask):
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
        from gan.util import add_gen_flag, normalize_per_sample

        sys.modules["output"] = output

        (data_feature, data_attribute,
         data_gen_flag,
         data_feature_outputs, data_attribute_outputs) = \
            load_data(os.path.join("..", "data", self._config["dataset"]))
        print(data_feature.shape)
        print(data_attribute.shape)
        print(data_gen_flag.shape)

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
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        sample_dir = os.path.join(self._work_dir, "sample")
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        time_path = os.path.join(self._work_dir, "time.txt")

        run_config = tf.compat.v1.ConfigProto()
        with tf.compat.v1.Session(config=run_config) as sess:
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
                num_packing=self._config["num_packing"],
                fix_feature_network=True)
            gan.build()

            if "restore" in self._config and self._config["restore"]:
                restore = True
            else:
                restore = False
            gan.train(
                restore=restore,
                feature_network_checkpoint_path=self._config["feature_network_checkpoint_path"])
