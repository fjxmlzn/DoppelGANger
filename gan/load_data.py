import os
import numpy as np
import pickle


def load_data(path, flag="train"):
    data_npz = np.load(
        os.path.join(path, "data_{}.npz".format(flag)))
    with open(os.path.join(path, "data_feature_output.pkl"), "rb") as f:
        data_feature_outputs = pickle.load(f)
    with open(os.path.join(path,
                           "data_attribute_output.pkl"), "rb") as f:
        data_attribute_outputs = pickle.load(f)

    data_feature = data_npz["data_feature"]
    data_attribute = data_npz["data_attribute"]
    data_gen_flag = data_npz["data_gen_flag"]
    return (data_feature, data_attribute,
            data_gen_flag,
            data_feature_outputs, data_attribute_outputs)
