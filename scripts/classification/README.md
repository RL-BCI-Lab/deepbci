
# Training and Testing
**All training and testing assumes the working directory is `../scripts/classification`.** 


## Training
Training is done by running via the `run_train.py` script. In order to run said script you will need to specify a model config and data config. The `configs/` directory contains examples of configs for both TensorFlow and Sklearn. 

 The model config is a yaml config file that defines the training parameters, model hyper-parameters, metrics, model pre and post processing, and other parameters specific to model. A local path to the model config should be passed to the `--model-cfg` argument. The data config is a yaml file that defines the data to be used for training/validation/testing and how to mutate said data. A local path to the data config should be passed to the `--data-cfg` argument. 

After training has finished, a directory in the `deep-bci/experiments/classification/exps` will be created which contains all the important meta-files needed for reproducing the training, results, training analysis,and logs/debugging. By default, if you didn't name the save directory when you ran `run_train.py` using the `--exp-dir` flag, `run_train.py` will create an experiment directory called `train-<date>-<time>` where the date and time are based on the current date/time.

### Running Training

To run a training example that does not require a GPU, you can run the following commands (be sure your working directory is `../deep-bci/experiments/classification/` when you run training).

```
python run_train.py --model-cfg configs/sklearn/svm/examples/trn.yaml --data-cfg configs/sklearn/svm/examples/trn-data.yaml
```

If you want to use different configs, simply change the above local paths to the `--model-cfg` and `--data-cfg` to new local paths within the `classification/` directory. Optionally, you can run the following command to see what other flags are available for use (e.g., caching data to prevent reloading the data every time).

```
python run_train.py --help 
```

## Testing
Testing is done by running the `run_test.py` script. In order to run said script you will need to specify a data config and model config, just like training, but now you also need to specify the experiment directory which contains all the training information.

Testing works slightly different from training in that it requires training, i.e. `run_train.py`, to have already been executed. This is because `run_train.py` creates a experiment directory where our training model and other meta-files were saved. Testing needs, at the very least, access to said trained model. Thus, in order for the `run_test.py` script, you must specify the `--exp-dir` flag. It should be noted you do not need to pass the absolute path to `--exp-dir` as the assumed working directory is `../deep-bci/scripts/classification`. 

### Running Testing

To run a testing example you can run the following command (be sure your working directory is ../deep-bci/experiments/classification/ when you run testing).

```
python run_test.py --model-cfg configs/sklearn/svm/examples/tst.yaml --data-cfg configs/sklearn/svm/examples/tst-data.yaml --exp-dir <name of directory generated during training with the exp/ directory>
```

If you want to use different configs, simply change the above local paths to the `--model-cfg` and `--data-cfg` to new local paths within the `classification/` directory.

Optionally, you can run the following command to see what other flags are available for use (e.g., caching training data).

```
python run_test.py --help
```

# Configs
Config files act as the instruction files for specifying which data to load and how to build the models you want to train. All config files will be compiled using [Hydra](https://hydra.cc/) which is a package that allows configs to define and instantiate objects directly from yaml files. Config files also utilize [OmegaConf](https://omegaconf.readthedocs.io/en/2.1_branch/index.html) which allows for arbitrary functions to be executed when the config is compiled as well (i.e., resolvers). This means, that you can run select functions from within the config, when it is compiled. This allows for information you might not know until runtime to be utilized.

## Model Config
You can see all the arguments you can specify for the model config by looking at the `TrainArgs` class located within `train.py` or the `TestArgs` class located within `test.py`. `TrainArgs` and `TestArgs` contain almost identical arguments although there are some differences. Each class contains further in-line documentation describing what the individual arguments are used for and if they are required or not. Additionally, many of the arguments take in dictionaries which means determining which values you can specify can be somewhat ambiguous. 

 **The key concept that all yaml configs follow in this repository is that they center around instantiation of objects or functions! This means that most of the configs will specify an object/function to instantiate using the `_target_` key and any keys located under the   `_target_` key act as kwargs to be passed to said object/function.** More on this in the next section.

### Resolvers
Resolvers are functions that will be executed automatically when the yaml file is loaded (these are called yaml resolvers) or when the config is instantiated using Hydra (these are [OmegaConf Resolvers](https://omegaconf.readthedocs.io/en/2.1_branch/custom_resolvers.html)). 

Below is a list of available resolvers you can utilize which are broken into two categories: yaml and OmegaConf.

- Yaml
  - `!eval` or `eval(<your input>)`: This will evaluate any string as Python code.
  - `!tuple` or `(<your input>)`: This will convert a list into a tuple as, by default, yaml can not create tuples.
  - `!none` or `None`: This will let you pass the value of None. By default, yaml uses the keyword "null" to pass None values

- OmegaConf
  - `${get: <your input>}`: Will get any callable method or function from the provided string import path.
  - `${cwd: <your input>}`: Builds an absolute path to the passed directory using your current working directory path ([docs](https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/)).
  - `${join: <your input>}`: Runs Python's 'join()' function which is used to combine a tuple or list of names into a file path using your OS's separator ([docs](https://docs.python.org/3/library/os.path.html#os.path.join))
  - `${get_data: <input one of the following: 'train'|'valid'|'test'>}`: Returns the data based on the passed split name. If you are training, you only have access to either the 'train' or 'valid' data (if valid data was even specified). If you are testing, you have have access to the 'test' data.
  - `${get_labels: <input one of the following: 'train'|'valid'|'test'>}`: Returns the labels based on the passed split name. If you are training, you only have access to either the 'train' or 'valid' labels (if valid data was even specified). If you are testing, you have have access to the 'test' labels.
  - `${get_exp_path: <input one of the following: 'train'|'test'>}`: Returns the path to the local 'train/' or 'test/' directories within your specified exp directory (`exp_dir`).
  - `${original_cwd:}`: Returns the original working directory.

### Example: Train 
Here we can see an example of a train model config. To see all the available parameters that can be passed to the model config and the description of the parameters see the `TrainArgs` located within the `train.py` script. 

Things of note:
- `_target_` is a Hyrda key that indicates an object/function needs to be instantiated. The value of the key corresponds to the import path of object/function (you can specify ANY import path as long as you have the package installed). Any arguments below the `_target_` argument will act as kwargs for building an instance of said object/function.
- `model_wrapper` is used to specify which library you want to build a model from.  In order to build a `model_warpper` you must specify a `_target_` key and the value should correspond to the import path to the desired `model_wrapper`. See the `deepbci/models/warppers` directory for all available model wrappers. Currently, this library supports Sklearn and TensorFlow wrappers. In the below example, we utilize the Sklearn wrapper `deepbci.models.wrappers.Sklearn`.
- `model` is used to specify the exact algorithm you want to use. Once again, this requires the utilization of `_target_` and a import path to the desired model. In the below example, we specify the import path to a support vector machine supplied by sklearn `sklearn.svm.SVC`.
- `dataset` contains kwargs for the dataset formating. Each `model_wrapper` has its own dataset class. For instance, TensorFlow utilizes TensorFlow datasets while Sklearn formats datasets using NumPy. This is the type the data will be converted into before training begins (regardless of any mutations you apply as well during preprocessing).
- `save` specifies the local save path within the specified experiment directory that is generated during training.
-  `${join:models,model.pickle}` is resolver code which will execute the `os.path.join` function using the 'models' and 'model.pickle' as inputs when the config is compiled.
- `evaluate_metrics` specifies which metrics to use for evaluation of the training results.
- `_partial_` is another Hyrda key that indicates the function being instatiated will not be ran. Instead it will partially declare the function using Python's [`partial()` function](https://www.geeksforgeeks.org/partial-functions-python/).

```
seed: 42

model_wrapper:
  _target_: deepbci.models.wrappers.Sklearn
  model:
    _target_: sklearn.svm.SVC
    C: 1.0
    kernel: linear
    class_weight: 
      _target_: deepbci.utils.class_weight.compute_class_weight
      y: ${get_labels:train}

dataset:
  shuffle: True
  
save:
  filepath: ${join:models,model.pickle}

evaluate_metrics:
  csv:
    bACC:
      _partial_: True
      _target_: sklearn.metrics.balanced_accuracy_score
    ACC:
      _partial_: True
      _target_: sklearn.metrics.accuracy_score
    tpr:
      _partial_: True
      _target_: sklearn.metrics.recall_score
    ppv:
      _partial_: True
      _target_: sklearn.metrics.precision_score
    tnr:
      _partial_: True
      _target_: sklearn.metrics.recall_score
      pos_label: 0
    F1: 
      _partial_: True
      _target_: sklearn.metrics.f1_score
    confusion_matrix:
      _partial_: True
      _target_: sklearn.metrics.confusion_matrix
  log:
    - confusion_matrix
  argmax_axis: 1
```

### Example: Test 
Here we can see an example of a test model config. To see all the available parameters that can be passed to the model config and the description of the parameters see the `TestArgs` located within the `test.py` script. 

Notice, that test simply requires you to specify the path of the model you want to load and `model_wrapper` will take care of the reset. 


```
model_wrapper:
  _target_: deepbci.models.wrappers.Sklearn
  load:
    filepath: ${join:${get_exp_path:train},models,model.pickle}

evaluate_metrics:
  csv:
    bACC:
      _partial_: True
      _target_: sklearn.metrics.balanced_accuracy_score
    ACC:
      _partial_: True
      _target_: sklearn.metrics.accuracy_score
    tpr:
      _partial_: True
      _target_: sklearn.metrics.recall_score
    ppv:
      _partial_: True
      _target_: sklearn.metrics.precision_score
    tnr:
      _partial_: True
      _target_: sklearn.metrics.recall_score
      pos_label: 0
    F1: 
      _partial_: True
      _target_: sklearn.metrics.f1_score
    confusion_matrix:
      _partial_: True
      _target_: sklearn.metrics.confusion_matrix
  log:
    - confusion_matrix
  argmax_axis: 1
```

## Data config
The data config format and parameters are slightly different then the model config. The data config is used to load all the datasets needed and the pre-process the datasets. There are two major keys for any given data config: groups and mutate. 

These two keys define the outer most dictionary. The dictionary with the key called 'groups' specifies all the data to be loaded. This data can be broken up into different groups using group names (i.e. sub-dictionaries, more on this later). The loaded data can then be mutated (changed in place). The dictionary with the key called 'mutate' specifies which data, the data you loaded using the 'groups' sub-dictionary, to mutate and how to mutate it. 


### Groups
The 'groups' dictionary in the data config is used by the `dbci.data_utils.data.Groups` class to load all the data into a Pandas.multi-index `Series`. The purpose of the `Groups` object is for storing different datasets while maintaining the ability to ID each dataset. Maintaining ID is important if you have multiple datasets that need to be pre-processed (mutated) differently. This means we need a way to quickly index and group datasets based on their IDs. This is role is fullfil by Pandas and multi-indexing.

The groups dictionary works as follows. The keys corresponds to the name of the group you wish to create. Groups essentially bind related data. These group names are in some instances arbitrary. However, the `run_train.py` and `run_test.py` scripts require specific group names to work correctly. The `run_train.py` script takes two group names: train, and valid. The 'train' group contains sub-dictionaries which specify all the data that will be loaded and used as the training data. The same idea applies to the 'valid' group. In addition, the `run_test.py` script requires a group named 'test'. Once again the same idea applies as before, all the data within the 'test' group will be used for testing.

The sub-dictionaries within a group name dictionary act as follows. The next layer of dictionary keys (right after the group names) correspond to `dbci.data_utils.data_loader` class names. These classes specify the dataset you want to load. The corresponding values are the kwargs for the `dbci.data_utils.data_loader` class's `load()` method that has been chosen. To view which datasets are available for loading and their required kwargs for the `load()` method checkout the `dbci.data_utils.data_loader` script. Any non-private classes (classes without the leading underscore `_`) can be loaded with no errors.

The classes you can load from `dbci.data_utils.data_loader` are given as follows:
- `BGSObs`: Loads the Observational Binary Goal Search task data.
- `BGSInt`: Loads the Interaction Binary Goal Search task data.
- `OAObs`: Loads the Observational Obstacle Avoidance task data.
- `OAOut`: Loads the Outcome Obstacle Avoidance task data.

### Mutate
The mutate dictionary works as follows. The mutate dictionary consists of a list which contains methods from the Groups class. These methods are used to mutate the data, as the Groups class purpose is for storing different datasets while maintaining each dataset ability to be IDed. The particular common methods used for pre-processing are the `apply_method()` and `apply_func()` methods.

**`Groups.apply_method()`**

This method works by calling methods available to the data object. To understand what this means it should be noted that all raw data and epoched data are stored via MNE objects. It is also possible to store data as NumPy arrays. This means you can use `apply_method()` to apply an MNE object method or NumPy array method. It is important to remember the type of your data. Further, if the method does not modify the data in-place then the method will not change the data. This can be overcome by implementing your own functions and using the `apply_func()` method instead.

**`Groups.apply_func()`**

This method works by calling functions from the `dbci.data_utils.mutators` file. There are various functions that a will modify the data in-place. If you want to add your own mutator functions simply add them to the `deepbci/data_utils/mutators.py` file.

### Example: Train 
The outer most dictionary keys for this config are 'groups' (required) and 'mutate' (optional). Within the groups dictionary are two named groups: train and valid. 

The train group loads trials 1-8 from the OAOut and OAObs datasets. Likewise, the valid group loads trial 9 from the OAOut dataset. 

In addition, wee can see that various mutate functions and methods are applied. For instance, the first mutator applies a bandpass filter. The next epochs the data which is followed by a compression. Compress is used to combine data. In this case all data up to the 'subject' level are compress (i.e., all trails are combined).

```
seed: 41

groups:
  _target_: deepbci.data_utils.Groups

  data_groups:
    train:
    - _target_: deepbci.data_utils.data_loaders.load_data
      data_loader: 
        _target_: deepbci.data_utils.data_loaders.OAOutLoader
      load_method: load_to_memory
      load_method_kwargs:
        subjects: [1]
        trials: eval(list(range(1, 7+1)))
        data_file: eeg.csv
        true_fs: false
        preload_epoch_indexes:
          generate_async_epochs:
            step_size: 100
            map_type: down
    - _target_: deepbci.data_utils.data_loaders.load_data
      data_loader: 
        _target_: deepbci.data_utils.data_loaders.OAOutLoader
      load_method: load_to_memory
      load_method_kwargs:
        subjects: [1]
        trials: [8]
        data_file: eeg.csv
        true_fs: false
        preload_epoch_indexes:
          generate_async_epochs:
            step_size: 100
            map_type: down
            
    valid:
    - _target_: deepbci.data_utils.data_loaders.load_data
      data_loader: 
        _target_: deepbci.data_utils.data_loaders.OAOutLoader
      load_method: load_to_memory
      load_method_kwargs:
        subjects: [1]
        trials: [9]
        data_file: eeg.csv
        true_fs: false
        preload_epoch_indexes:
          generate_async_epochs:
            step_size: 100
            map_type: down

mutate:

- apply_func:
    select: null
    func:
      _partial_: True
      _target_: deepbci.data_utils.mutators.filter
      l_freq: 0.1
      h_freq: 30
      method: iir
      verbose: False
      iir_params:
        order: 4
        ftype: butter
        output: sos

- apply_func:
    select: null
    func:
      _partial_: True
      _target_: deepbci.data_utils.mutators.epoch
      tmin: 0.1
      tmax: 0.495
      preload: true
      picks: [eeg]
      verbose: WARNING
      baseline: null

- compress:
    compress_level: subject
    select: [train]

- apply_func:
    select: null
    func:
      _partial_: True
      _target_: deepbci.data_utils.mutators.to_numpy
      units: uV

- compress:
    compress_level: group

- apply_func:
    select: [train]
    func:
      _partial_: True
      _target_: deepbci.data_utils.mutators.rescale
      scaler:
        _target_: deepbci.data_utils.scalers.STD
        axis: [0,2]
      save: true 
      filepath: scalers.pickle
- apply_func:
    select: [valid]
    func:
      _partial_: True
      _target_: deepbci.data_utils.mutators.rescale
      load: true 
      filepath: scalers.pickle

- apply_func:
    select: null
    func: 
      _partial_: True
      _target_: deepbci.data_utils.mutators.compress_dims
      start: 1 
```

### Example: Test 
The test config is very similar to the train config with the main difference being the group name is now 'test'.

```
seed: 41

groups:
  _target_: deepbci.data_utils.Groups
  data_groups:
    test:
    - _target_: deepbci.data_utils.data_loaders.load_data
      data_loader: 
        _target_: deepbci.data_utils.data_loaders.OAOutLoader
      load_method: load_to_memory
      load_method_kwargs:
        subjects: [1]
        trials: [9, 10]
        data_file: eeg.csv
        true_fs: false
        preload_epoch_indexes:
          generate_async_epochs:
            step_size: 100
            map_type: down
    - _target_: deepbci.data_utils.data_loaders.load_data
      data_loader: 
        _target_: deepbci.data_utils.data_loaders.OAObsLoader
      load_method: load_to_memory
      load_method_kwargs:
        subjects: [1]
        trials: eval(list(range(1, 10+1)))
        data_file: eeg.csv
        true_fs: false
        preload_epoch_indexes:
          generate_async_epochs:
            step_size: 100
            map_type: down

mutate:

- apply_func:
    select: null
    func:
      _partial_: True
      _target_: deepbci.data_utils.mutators.filter
      l_freq: 0.1
      h_freq: 30
      method: iir
      verbose: False
      iir_params:
        order: 4
        ftype: butter
        output: sos

- apply_func:
    select: null
    func:
      _partial_: True
      _target_: deepbci.data_utils.mutators.epoch
      tmin: 0.1
      tmax: 0.495
      preload: true
      picks: [eeg]
      verbose: WARNING
      baseline: null

- apply_func:
    select: null
    func:
      _partial_: True
      _target_: deepbci.data_utils.mutators.to_numpy
      units: uV

- apply_func:
    select: [test]
    func:
      _partial_: True
      _target_: deepbci.data_utils.mutators.rescale
      load: true 
      filepath: scalers.pickle

- apply_func:
    select: null
    func: 
      _partial_: True
      _target_: deepbci.data_utils.mutators.compress_dims
      start: 1
```

# Running multi-experiments
*Implemented but lacking documentation*

# Running cross-validation
*Implemented but lacking hyper-parameter tuning and documentation*