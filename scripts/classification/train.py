import os
from os.path import join
from pdb import set_trace
from datetime import datetime
from dataclasses import dataclass, asdict, field, is_dataclass

import dill
import numpy as np
import pandas as pd
from hydra.utils import instantiate, get_original_cwd, to_absolute_path, get_method
from omegaconf import OmegaConf

import deepbci as dbci
import deepbci.utils.utils as utils
import deepbci.utils.logger as logger
from deepbci.data_utils.data import Groups, run_group_mutators

import directory_structure as ds
from utils import set_all_seeds

def main(model_args, exp_dir=None, data_cfg=None, load_cache=None, cache=None):    
    # Create save directory i.e. experiment path
    if exp_dir is None:
        now = datetime.now()
        exp_dir = "train-{}".format(now.strftime("%Y-%m-%d-%H-%M-%S"))

    exp_path = build_exp_path(exp_dir=exp_dir)
    trn_exp_path = join(exp_path, ds.TRN_DIR)
    if not os.path.exists(trn_exp_path):
        os.makedirs(trn_exp_path)
        
    original_dir = os.getcwd()
    os.chdir(exp_path)
    
    logger.configure('logs',  ['log'], log_prefix='train-main-')
    main_logger = logger.Logger.CURRENT
    ####################################################################################
    # Data loading (building Groups object from configs)
    ####################################################################################
    # Check if load_cache is passed, if not, check for data_cfg path. 
    # Case 1: If both are not passed then error is thrown. 
    # Case 2: If both are passed then the cached data is used.
    set_default_resolvers()
    if load_cache is not None:
        grps, data_cfg = cache_load(load_cache)
    elif data_cfg is not None:
        built_data_cfg = instantiate_data_config(data_cfg, seed=model_args['seed'])
        grps = mutate_groups(
            grps=built_data_cfg['groups'], 
            mutate=built_data_cfg.get('mutate'), 
            seed=model_args['seed']
        )
    else:
        err = "data_cfg and load_cache are both None, be sure to provide one or the other."
        raise RuntimeError(err)
 
    # Cache grps object and data_cfg for faster loading 
    if cache is not None:
        cache_save(grps, data_cfg, cache)
    
    # Save configs
    save_configs(data_cfg=data_cfg, model_args=model_args, prefix="trn")

    # Extract train and validation multi-index Dataframes from Groups object
    trn_df, vld_df = get_train_valid_group(grps)
    
    # Extract data from Multi-Index DataFrame
    trn = trn_df.ravel()[0]
    vld = vld_df.ravel()[0] if vld_df is not None else None
    
    tag_headers = list(trn_df.index.names)
    trn_tags = trn_df.index.unique()[0]
    vld_tags = vld_df.index.unique()[0] if vld_df is not None else None

    train(
        trn=trn,
        trn_tags=trn_tags,
        tag_headers=tag_headers,
        vld=vld, 
        vld_tags=vld_tags,
        trn_exp_path=trn_exp_path, 
        model_args=model_args
    )
    
    # Revert back to original working directory
    os.chdir(original_dir)
    
def build_exp_path(exp_dir, make_dir=True):
    """ Build save directory using passed directory name

        This directory will always be built within the EXPS_DIR specified at
        the top of this script.
        
        model_args:
            exp_dir (str): Name of the directory where models, metrics, and logs will
                be saved.
        
    """
    
    if ds.EXPS_DIR in exp_dir:
        exp_path = join(ds.ROOT_DIR, exp_dir)
    else:
        exp_path = join(ds.ROOT_DIR, ds.EXPS_DIR, exp_dir)
  
    if make_dir and not os.path.exists(exp_path):
        os.makedirs(exp_path)
        
    return exp_path

def mutate_groups(grps, mutate=None, seed=None):
    """ Apply mutators specified by mutator config to preprocess data in Groups object.
        
        Args:
            gros (Groups): deepbci.data_utils.Groups object
            
            mutate (dict): List of dicts that specify how Groups.data_map should be 
                mutated.
                
        Returns:
            Returns same Groups object. However, all changes are done in-place.
    """
    # Set seeds for any mutations that use random processes
    set_all_seeds(seed)

    # Apply data preprocessing pipeline (mutators) to prep data for training
    if mutate is not None:
        run_group_mutators(grps, mutate)

    return grps

def get_train_valid_group(grps):
    """ Extracts train and validation groups from deepbci.data_utils.data.Groups

        model_args:
            grps (deepbci.Groups): Groups object that contains train and validation
                data splits.
        Note:
            Be sure when creating group names that a group is named 'train'.
            This will act as the training data. Likewise, a validation group name 'valid' 
            can be provided, this will act as the validation data. Both of these groups must
            be compiled down to a single deepbci.data_utils.DataGroup object to be used properly.
            If you do not do so then the first DataGroup object in the group will be used.
    """

    trn_df = grps.data_map.loc[['train']]
    if 'valid' in grps.get_level_values(level='group').unique():
        vld_df = grps.data_map.loc[['valid']]
    else:
        vld_df = None

    return trn_df, vld_df
    
def cache_save(data, data_cfg, cache_dir):
    """ Cache data via dill
        
        Instead of building data with a data_cfg it can be faster to load a cached
        pre-built dataset. Use the command line arg --cache to indicate caching of current
        data_cfg. The default file location is ./cache/data-cache. However, you can pass
        a argument to --cache which will specify the cache filename.
        
        model_args:
            data (list, tuple): Contains objects to be cached.
            
            cache_dir (str): Name of dir to cache data to within 
                scripts/classification/cache/.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    data_cfg_file = join(cache_dir, ds.DATA_CFG)
    utils.dump_yaml(data_cfg, data_cfg_file, default_flow_style=None, sort_keys=False)

    data_file = join(cache_dir, ds.CACHE_FILE)
    with open(data_file, 'wb') as f:
        dill.dump(data, f)

def cache_load(cache_dir):
    """ Load cached data via dill.
        
        model_args:
            cache_dir (str): Name of dir to load cached data from. This file must be
                within  scripts/classification/cache/ directory.
    """ 
    data_cfg_file = join(cache_dir, ds.DATA_CFG)
    data_cfg = utils.load_yaml(data_cfg_file)

    data_file = join(cache_dir, ds.CACHE_FILE)
    with open(data_file, 'rb') as f:
       data = dill.load(f)

    return data, data_cfg

def train(
    trn, 
    trn_tags,
    tag_headers,
    model_args, 
    trn_exp_path, 
    vld=None, 
    vld_tags=None, 
    prebuilt=None
):
    """ Main code for training a model """  
    check_tags(trn_tags, tag_headers, vld_tags=vld_tags)
    
    # Change working directory and save old working directory 
    original_dir = os.getcwd()
    os.chdir(trn_exp_path)
    # Initialize logger
    og_logger = logger.Logger.CURRENT
    logger.configure('logs',  ['log'], log_prefix='train-')
    # trn = trn_df.ravel()[0]
    # vld = vld_df.ravel()[0] if vld_df is not None else None
    metrics = {}

    # Reset all seeds for training
    seed = model_args['seed']
    logger.log(f"Setting all seeds to {seed}")
    set_all_seeds(seed)
    
    ####################################################################################
    # Instantiate config and extract model wrapper
    ####################################################################################
    logger.info("Instantiating model config...")
    set_default_model_resolvers(trn=trn, vld=vld, trn_exp_path=trn_exp_path)
    model_args = instantiate_model_config(args_class=TrainArgs, 
                                          model_args=model_args, 
                                          prebuilt=prebuilt)
    logger.info("SUCCESS: Model config instantiated")
    model_wrapper = model_args['model_wrapper']

    # Get the model wrapper's required dataset class and format data
    # using the class stored within dataset
    trn_set = (trn.data, trn.labels)
    vld_set = (vld.data, vld.labels) if vld is not None else None
    dataset = model_wrapper.dataset(trn_set=trn_set, vld_set=vld_set, **model_args['dataset'])

    ####################################################################################
    # Model training, prediction, and evaluation
    ####################################################################################
    # Compute class weights by default. However, passing classweights to fit model_args will
    # override compute class weights.
    logger.info("Training model...")
    train_timer = utils.Timer()
    train_timer.start()
    model_wrapper.fit(dataset.trn_set, dataset.vld_set, **model_args['fit'])
    train_time = train_timer.stop()
    logger.info(f"SUCCESS: Training complete. Time to complete {train_time}")
    
    # Test only unique training sample. Results are distorted when using upsampling
    # as it adds duplicate data! 
    _, unique_indices = np.unique(trn.data, axis=0, return_index=True)
    if len(unique_indices) != len(trn.data):
        logger.warn(f"Duplicate training data samples found: " \
                   f"Unique {len(unique_indices)} Total: {len(trn.data)}. "
                   f"Using unique samples only for evaluation.")
        unique_indices.sort()
        trn.labels = trn.labels[unique_indices]
        trn.data = trn.data[unique_indices]
        dataset.trn_set = (trn.data, trn.labels)

    # Set training shuffling to false
    dataset.shuffle = False
    # Get training predictions and metrics
    logger.info("Computing TRAINING predictions and metrics...")
    trn_preds = model_wrapper.predict(dataset.trn_set, **model_args['predict'])
    trn_metrics = evaluate_metrics(trn.labels, trn_preds, **model_args['evaluate_metrics'])
    # Save metrics paired with group name train
    metrics[trn_tags] = trn_metrics
    # metrics[trn_df.index.unique()[0]] = trn_metrics
    logger.info("SUCCESS: Predictions and metrics computed")

    # Get validation predictions and metrics
    if vld is not None:
        logger.info("Computing VALIDATION predictions and metrics...")
        vld_preds = model_wrapper.predict(dataset.vld_set, **model_args['predict'])
        vld_metrics = evaluate_metrics(vld.labels, vld_preds, **model_args['evaluate_metrics'])
        # Save metrics paired with group name valid
        metrics[vld_tags] = vld_metrics
        # metrics[vld_df.index.unique()[0]] = vld_metrics
        logger.info("SUCCESS: Predictions and metrics computed")

    ####################################################################################
    # Result/statistics logging 
    ####################################################################################
   
    # Save metrics
    logger.info("Saving metrics...")
    save_metrics(metrics=metrics, multi_idx_names=tag_headers)
    # save_metrics(metrics=metrics, multi_idx_names=trn_df.index.names)
    logger.info("SUCCESS: Metric saved")
    
    # Save train predictions 
    # save_preds(
    #     file_name=ds.TRN_PRED_FILE,
    #     preds=trn_preds,
    #     labels=trn.labels
    # )

    # Save valid predictions
    # if vld is not None:
    #     save_preds(
    #         file_name=ds.VLD_PRED_FILE,
    #         preds=vld_preds,
    #         labels=vld.labels
    #     )
    if model_args['save']:
        logger.info("Saving model...")
        model_wrapper.save(**model_args['save'])
        logger.info("SUCCESS: Model saved")
        
    # Revert to original working directory
    logger.Logger.CURRENT = og_logger
    os.chdir(original_dir)

    return model_args, metrics 

def check_tags(trn_tags, tag_headers, vld_tags=None):
    if not isinstance(trn_tags, tuple):
        err = f"'trn_tags' must be a tuple of strings: received {type(trn_tags)}"
        raise ValueError(err)
    if not isinstance(tag_headers, (tuple, list, np.ndarray)):
        err = f"'tag_headers' must be a tuple, list, or np.ndarray of strings: received {type(tag_headers)}"
        raise ValueError(err)
    if vld_tags is not None and not isinstance(vld_tags, tuple):
        err = f"'vld_tags' must be a tuple of strings: received {type(vld_tags)}"
        raise ValueError(err)

def instantiate_data_config(data_cfg, seed=None):
    # Set seeds incase data loading has some random processes
    set_all_seeds(seed)
    # Instantiate objects
    built_data_cfg = instantiate(data_cfg, _convert_='all')
    
    return built_data_cfg

def get_matching_keys(keys, target_keys):
    keys = set(keys)
    target_keys = set(target_keys)
    return keys.intersection(target_keys)
    
def instantiate_model_config(args_class, model_args, prebuilt=None):
    prebuilt = {} if prebuilt is None else prebuilt
    if is_dataclass(prebuilt):
        prebuilt = prebuilt.__dict__
    if is_dataclass(model_args):
         model_args = model_args.__dict__
    
    model_module = model_args['model_wrapper'].get('_target_')
    if model_module is not None:
        preinstantiate_model_wrapper(model_module=model_module, **model_args['preinstantiate'])
    else:
        logger.warn("model_wrapper config has no key `_target_`. No object will be built!")
        
    matched_keys = get_matching_keys(model_args.keys(), prebuilt.keys())
    if len(matched_keys) > 0:
        logger.info(f"Excluding keys that have already been prebuilt when instantiating model_args: {matched_keys}")
        model_args = {key:value for key, value in model_args.items() if key not in matched_keys}
        prebuilt = {key:prebuilt[key] for key in matched_keys}
    else:
        logger.info(f"Using all keys when instantiating model_args")
        
    partially_built = instantiate(model_args, _convert_='all')
    built_model_args = args_class(**partially_built, **prebuilt)
    
    return built_model_args.__dict__

def set_default_resolvers():
    OmegaConf.register_new_resolver('get', get_method, replace=True)
    OmegaConf.register_new_resolver('cwd', to_absolute_path, replace=True)
    OmegaConf.register_new_resolver('join', join, replace=True)

def set_default_model_resolvers(
    trn=None, 
    vld=None, 
    tst=None, 
    trn_exp_path=None, 
    tst_exp_path=None
):
    
    def get_data(split):
        if split == 'train' and trn is not None:
            data = trn.data
        elif split == 'valid' and vld is not None:
            data = vld.data
        elif split == 'test' and tst is not None:
            data = tst.data
        else:
            raise ValueError(f"The data split {split} does not exist")
        
        return data
    
    def get_labels(split, argmax_axis=None):
        if split == 'train' and trn is not None:
            labels = trn.labels
        elif split == 'valid' and vld is not None:
            labels = vld.labels
        elif split == 'test' and tst is not None:
            labels = tst.labels
        else:
            raise ValueError(f"The label split {split} does not exist")
        
        if argmax_axis is not None and len(labels.shape) > 1:
            labels = np.argmax(labels, axis=argmax_axis)
            
        return labels
    
    def get_exp_path(stage):
        if stage == 'train' and trn_exp_path is not None:
            return trn_exp_path
        elif stage == 'test' and tst_exp_path is not None:
            return tst_exp_path
        else:
            raise ValueError(f"The stage {stage} does not have an experiment path")
    
    OmegaConf.register_new_resolver('get_data', get_data, replace=True)
    OmegaConf.register_new_resolver('get_labels', get_labels, replace=True)
    OmegaConf.register_new_resolver('get_exp_path', get_exp_path, replace=True)
    OmegaConf.register_new_resolver('original_cwd', get_original_cwd, replace=True)

def preinstantiate_model_wrapper(model_module: str, **kwargs):
    model_wrapper = utils.import_module(model_module)
    if hasattr(model_wrapper, 'preinstantiate'):
        model_wrapper.preinstantiate(**kwargs)

def evaluate_metrics(y_true, y_pred, csv=None, log=None, exclude_argmax=None, argmax_axis=None):
    """ Adds extra metrics that were not computed during training 
    
        model_args:
            preds (nd.array): Predictions from model.
            
            labels (np.array): True labels for data.
    """
    metrics = {}
    csv = {} if csv is None else csv
    log = [] if log is None else log
    exclude_argmax = [] if exclude_argmax is None else exclude_argmax

    if len(y_true.shape) > 1 and argmax_axis is not None:
        y_true_1d = np.argmax(y_true, axis=argmax_axis).reshape(-1, 1)
    else:
         y_true_1d = y_true
    if len(y_pred.shape) > 1 and argmax_axis is not None:
        y_pred_1d = np.argmax(y_pred, axis=argmax_axis).reshape(-1, 1)
    else:
        y_pred_1d = y_pred
        
    for name, metric in csv.items():
        if name in exclude_argmax:
            score = metric(y_true, y_pred)
        else:
            score = metric(y_true_1d, y_pred_1d)

        if name in log:
            logger.info(f"{name}:\n{score}")
        else:
            metrics[name] = score
    
    return metrics
        
def save_configs(data_cfg=None, model_args=None, prefix=''):
    """ Save configs to be reused for replication.
    
        model_args:
            data_cfg (dict): Data config file for constructing training and validation
                data. Should be unedited so that using this saved data config can be
                reused.
                
            model_args (dict): Model config file for building model. Should be unedited 
                so that using this saved model config can be reused.
    """
    if data_cfg is not None or model_args is not None:
        if not os.path.exists(ds.CFG_DIR):
            os.makedirs(ds.CFG_DIR)

    if data_cfg:
        data_cfg_file = f"{prefix}-{ds.DATA_CFG}"  if len(prefix) > 0 else ds.DATA_CFG
        data_cfg_path = join(ds.CFG_DIR, data_cfg_file)
        utils.dump_yaml(data_cfg, data_cfg_path, default_flow_style=None, sort_keys=False)
    
    if model_args:
        model_args_file = f"{prefix}-{ds.MODEL_CFG}" if len(prefix) > 0 else ds.MODEL_CFG
        model_args_path = join(ds.CFG_DIR, model_args_file)
        utils.dump_yaml(model_args, model_args_path, default_flow_style=None, sort_keys=False)

def save_metrics(metrics, multi_idx_names, metrics_file=None):
    """ Save metrics and summaries for every sub-experiment.
    
        model_args:       
            metrics (dict): Metrics to be saved where the key corresponds to the metric
                name and the value corresponds to the metric value. 
    """  
    metrics_file = ds.METRICS_FILE if metrics_file is None else metrics_file
    
    results_dir = ds.RESULTS_DIR
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    multi_idx = pd.MultiIndex.from_tuples(metrics.keys(), names=multi_idx_names)
    metrics = pd.DataFrame(metrics.values(), index=multi_idx)
    metrics_file = join(results_dir, metrics_file)
    metrics.to_csv(metrics_file, mode='w', header=True)
    
    return metrics

# def save_preds(file_name, preds, labels, timestamps=None):
#     """ Save probabilities, predictions, and labels.
    
#         model_args:    
#             file_name (str): Name of file to be saved to.
                        
#             preds (nd.array): Predictions from model.
            
#             labels (np.array): True labels for data.
            
#     """
#     # Add class column names
#     class_tmpl = "class {}"
#     classes = []
#     for c in np.unique(labels):
#         classes.append(class_tmpl.format(c))
#     columns = classes + ['prediction', 'label']

#     # Add timestamp column name if given
#     if timestamps is not None:
#         columns += ['timestamp']

#     # Build directory path
#     if not os.path.exists(ds.PREDS_DIR):
#         os.makedirs(ds.PREDS_DIR)
        
#     pred_labels = np.argmax(preds, axis=1).reshape(-1, 1)
#     pred_file = join(ds.PREDS_DIR, file_name)
#     df = pd.DataFrame(np.hstack([preds, pred_labels, labels]), columns=columns)
#     df.to_csv(pred_file, mode='w', index=False)
    
#     return preds

@dataclass(order=True)
class _TrainRequiredArgs():
    """
        model wrapper (dict): Defines which module wrapper to use. Specify module path 
                to a model wrapper using the key '_target_'. Any other keys with this 
                dictionary will act as kwarg for instantiating the chosen wrapper class. 
    """ 
    model_wrapper: dict
    
@dataclass(order=True)
class _TrainDefaultArgs():
    """ Dataclass that stores variables that will be used by the run_train.py script.

        Attributes:    
            predict (dict): Kwargs for the model wrapper's test() method
            
            fit (dict): Kwargs for the model wrapper's fit() method

            scaler (dict):  Defines standardization/normalization algorithm to use.
                Specify module path to scaler using the key '_target_'. Any other
                keys with this dictionary will act as kwarg for instantiating the scaler
                class chosen. 
            
            seed (int): Seed to use for any random processes (e.g., during mutation or training)
            
            dataset (dict): Kwargs for the model wrapper's dataset class.
            
            preinstantiate (dict): Kwargs for the model wrapper's preinstantiate() method.


    """
    predict: dict = field(default_factory=dict)
    fit: dict = field(default_factory=dict)
    preinstantiate: dict = field(default_factory=dict)
    seed: int = np.random.randint(99999)
    dataset: dict = field(default_factory=dict)
    evaluate_metrics: dict = field(default_factory=dict)
    save: dict = None
        
    def __getitem__(self, item):
        return getattr(self, item)
    
@dataclass(order=True)
class TrainArgs(_TrainDefaultArgs, _TrainRequiredArgs):
    """ Class which combines both _TrainDefaultArgs and _TrainRequiredArgs"""
        
    def __getitem__(self, item):
        return getattr(self, item)