config = {
    "scheduler_config": {
        "gpu": ["0"],
        "config_string_value_maxlen": 1000,
        "result_root_folder": "../results/"
    },

    "global_config": {
        "batch_size": 100,
        "vis_freq": 200,
        "vis_num_sample": 5,
        "d_rounds": 1,
        "g_rounds": 1,
        "num_packing": 1,
        "noise": True,
        "feed_back": False,
        "g_lr": 0.001,
        "d_lr": 0.001,
        "d_gp_coe": 10.0,
        "gen_feature_num_layers": 1,
        "gen_feature_num_units": 100,
        "gen_attribute_num_layers": 3,
        "gen_attribute_num_units": 100,
        "disc_num_layers": 5,
        "disc_num_units": 200,
        "initial_state": "random",

        "attr_d_lr": 0.001,
        "attr_d_gp_coe": 10.0,
        "g_attr_d_coe": 1.0,
        "attr_disc_num_layers": 5,
        "attr_disc_num_units": 200,
    },

    "test_config": [
        {
            "dataset": ["web"],
            "epoch": [15],
            "run": [0],
            "sample_len": [10],
            "extra_checkpoint_freq": [5],
            "epoch_checkpoint_freq": [1],
            "aux_disc": [True],
            "self_norm": [True],
            "dp_noise_multiplier": [0.01, 0.1, 1.0, 2.0, 4.0],
            "dp_l2_norm_clip": [1.0]
        }
    ]
}
