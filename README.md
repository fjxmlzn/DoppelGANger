# Generating High-fidelity, Synthetic   Time Series Datasets with DoppelGANger

**[[paper (arXiv)]()]**
**[[code](https://github.com/fjxmlzn/DoppelGANger)]**


**Authors:** [Zinan Lin](http://www.andrew.cmu.edu/user/zinanl/), [Alankar Jain](https://www.linkedin.com/in/alankar-jain-5835ab5a/), [Chen Wang](https://wangchen615.github.io/), [Giulia Fanti](https://www.andrew.cmu.edu/user/gfanti/), [Vyas Sekar](https://users.ece.cmu.edu/~vsekar/)

**Abstract:** Limited data access is a substantial barrier to data-driven networking research and development. Although many organizations are motivated to share data, privacy concerns often prevent the sharing of proprietary data, including between teams in the same organization and with outside stakeholders (e.g., researchers, vendors). Many researchers have therefore proposed synthetic data models, most of which have not gained traction because of their narrow scope. In this work, we present DoppelGANger, a synthetic data generation framework based on generative adversarial networks (GANs). DoppelGANger is designed to work on time series datasets with both continuous features (e.g. traffic measurements) and discrete ones (e.g., protocol name). Modeling time series and mixed-type data is known to be difficult; DoppelGANger circumvents these problems through a new conditional architecture that isolates the generation of metadata from time series, but uses metadata to strongly influence time series generation. We demonstrate the efficacy of DoppelGANger on three real-world datasets. We show that DoppelGANger achieves up to 43% better fidelity than baseline models, and captures structural properties of data that baseline methods are unable to learn. Additionally, it gives data holders an easy mechanism for protecting attributes of their data without substantial loss of data utility. 

---
This repo contains the codes of DoppelGANger. The codes were tested under Python 2.7.5, TensorFlow 1.4.0.

## Dataset format
To train DoppelGANger for your data, you need to prepare your data  according to the following format, which contains three files:

* `data_feature_output.pkl`: A pickle dump of a list of `gan.output.Output` objects, indicating the dimension, type, normalization of each feature.
* `data_attribute_output.pkl`: A pickle dump of a list of `gan.output.Output` objects, indicating the dimension, type, normalization of each attribute.
* `data_train.npz`: A numpy `.npz` archive of the following three arrays:
	* `data_feature`: Training features, in numpy float32 array format. The size is [(number of training samples) x (maximum length) x (total dimension of features)]. Categorical features are stored by one-hot encoding; for example, if a categorical feature has 3 possibilities, then it can take values between `[1., 0., 0.]`, `[0., 1., 0.]`, and `[0., 0., 1.]`. Each continuous feature should be normalized to `[0, 1]` or `[-1, 1]`. The array is padded by zeros after the time series ends.
	* `data_attribute`: Training attributes, in numpy float32 array format. The size is [(number of training samples) x (total dimension of features)]. Categorical attributes are stored by one-hot encoding; for example, if a categorical attribute has 3 possibilities, then it can take values between `[1., 0., 0.]`, `[0., 1., 0.]`, and `[0., 0., 1.]`. Each continuous attribute should be normalized to `[0, 1]` or `[-1, 1]`.
	* `data_gen_flag`: Flags indicating the activation of features, in numpy float32 array format. The size is [(number of training samples) x (maximum length)]. 1 means the time series is activated at this time step, 0 means the time series is inactivated at this timestep. 

Let's look at a concrete example. Assume that there are two features (a 1-dimension continuous feature normalized to [0,1] and a 2-dimension categorical feature) and two attributes (a 2-dimension continuous attribute normalized to [-1, 1] and a 3-dimension categorical attributes). Then `data_feature_output ` and `data_attribute_output ` should be:

```
data_feature_output = [
	Output(type_=CONTINUOUS, dim=1, normalization=ZERO_ONE, is_gen_flag=False),
	Output(type_=DISCRETE, dim=2, normalization=None, is_gen_flag=False)]
	
data_attribute_output = [
	Output(type_=CONTINUOUS, dim=2, normalization=MINUSONE_ONE, is_gen_flag=False),
	Output(type_=DISCRETE, dim=3, normalization=None, is_gen_flag=False)]
```

Note that `is_gen_flag` should always set to `False` (default). `is_gen_flag=True` is for internal use only (see comments in `doppelganger.py` for details).

Assume that there are two samples, whose lengths are 2 and 4, and assume that the maximum length is set to 4. Then `data_feature `, `data_attribute `, and `data_gen_flag ` could be:

```
data_feature = [
	[[0.2, 1.0, 0.0], [0.4, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
	[[0.9, 0.0, 1.0], [0.3, 0.0, 1.0], [0.2, 0.0, 1.0], [0.8, 1.0, 0.0]]]
	
data_attribute = [
	[-0.2, 0.3, 1.0, 0.0, 0.0],
	[0.2, 0.3, 0.0, 1.0, 0.0]]
	
data_gen_flag = [
	[1.0, 1.0, 0.0, 0.0],
	[1.0, 1.0, 1.0, 1.0]]
```

The datasets we used in the paper (Wikipedia Web Traffic, Google Cluster Usage Traces, Measuring Broadband America) can be found [here](https://drive.google.com/drive/folders/19hnyG8lN9_WWIac998rT6RtBB9Zit70X?usp=sharing).

## Run DoppelGANger
The codes are based on [GPUTaskScheduler](https://github.com/fjxmlzn/GPUTaskScheduler) library, which helps you automatically schedule jobs among GPU nodes. Please install it first. You may need to change GPU configurations according to the devices you have. The configurations are set in `config*.py` in each directory. Please refer to [GPUTaskScheduler's GitHub page](https://github.com/fjxmlzn/GPUTaskScheduler) for details of how to make proper configurations.

> You may also run these codes without GPUTaskScheduler. See the \_\_main\_\_ section in `doppelganger.py` for an example.

The implementation of DoppelGANger is at `gan/doppelganger.py`. You may refer to the comments in it for details. Here we provide our code for training DoppelGANger on the three datasets (Wikipedia Web Traffic, Google Cluster Usage Traces, Measuring Broadband America) in the paper, and give examples on using DoppelGANger to generate data and retraining the attribute generation network.

### Download dataset
Before running the code, please download the three datasets [here](https://drive.google.com/drive/folders/19hnyG8lN9_WWIac998rT6RtBB9Zit70X?usp=sharing) and put it under `data` folder.

### Train DoppelGANger
```
cd example_training
python main.py
```

### Generate data by DoppelGANger
```
cd example_generating_data
python main_generate_data.py
```

### Retrain attribute generation network of DoppelGANger
Put your data with the desired attribute distribution in `data/web_retraining`, and then

```
cd example_retraining_attribute
python main.py
```
### Customize DoppelGANger
You can play with the configurations (e.g., whether to have the auxiliary discriminator) in `config*.py`.