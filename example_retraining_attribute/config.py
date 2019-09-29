config = {
    "scheduler_config": {
        "gpu": ["0"],
        "config_string_value_maxlen": 1000,
        "result_root_folder": "../results_retraining/",
        "temp_folder": "../temp/"
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

        "feature_network_checkpoint_path":"../results/aux_disc-True,dataset-web,epoch-400,epoch_checkpoint_freq-1,extra_checkpoint_freq-5,run-0,sample_len-10,self_norm-True,/checkpoint/epoch_id-399/model-199999"
    },

    "test_config": [
        {
            "dataset": ["web_retraining"],
            "epoch": [100],
            "run": [0],
            "sample_len": [10],
            "extra_checkpoint_freq": [5],
            "epoch_checkpoint_freq": [1],
            "aux_disc": [True],
            "self_norm": [True]
        }
    ]
}
