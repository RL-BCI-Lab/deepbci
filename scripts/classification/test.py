import os
import re 
from os.path import join
from dataclasses import dataclass, asdict, field
from pdb import set_trace

import dill
import numpy as np
import pandas as pd

import deepbci as dbci
import deepbci.utils.utils as utils
import deepbci.utils.logger as logger
import deepbci.utils.compress as compress
from deepbci.data_utils.data import Groups, run_group_mutators

import train
import directory_structure as ds 

def main(model_args, exp_dir, data_cfg=None, load_cache=None, cache=None):
    exp_path = train.build_exp_path(exp_dir, make_dir=False)
    tst_exp_path = join(exp_path, ds.TST_DIR)
    if not os.path.exists(tst_exp_path):
        os.makedirs(tst_exp_path)
      
    original_dir = os.getcwd()
    os.chdir(exp_path)
    
    logger.configure('logs',  ['log'], log_prefix='test-main-')
    main_logger = logger.Logger.CURRENT
    
    ####################################################################################
    # Data loading, splitting and caching
    ####################################################################################
    # Check if load_cache is passed, if not, check for data_cfg path. If both are not
    # passed then error is thrown. If both are passed then the cached data is used.
    train.set_default_resolvers()
    if load_cache is not None:
        grps, data_cfg = cache_load(load_cache)
    elif data_cfg is not None:
        built_data_cfg = train.instantiate_data_config(data_cfg, seed=model_args['seed'])
        grps = train.mutate_groups(
            grps=built_data_cfg['groups'], 
            mutate=built_data_cfg.get('mutate'), 
            seed=model_args['seed']
        )
    else:
        err = "data_cfg and load_cache are both None, be sure to provide one or the other."
        raise RuntimeError(err)
    
    # Create cache for current data
    if cache is not None:
        train.cache_save(grps, data_cfg, cache)

    # Save configs
    train.save_configs(data_cfg=data_cfg, model_args=model_args, prefix="tst")
    
    tst_df = get_test_group(grps)
    tst = tst_df.ravel()
    tag_headers = list(tst_df.index.names)
    tst_tags = tst_df.index.unique()
    
    test(
        tst=tst,
        tst_tags=tst_tags,
        tag_headers=tag_headers,
        tst_exp_path=tst_exp_path, 
        model_args=model_args
    )
    
    # Revert back to original working directory
    os.chdir(original_dir)

def get_test_group(grps):
    """ Returns test group from the deepbci.data_utils.data.Groups object

        Note:
            Be sure that the group name 'test' is included. Data contained within 
            this group will act as the testing data. Note, data can be compiled 
            at any level. Unlike training, testing can be an array of data objects. 
            No compiling is even needed if you want to run tests at the trial 
            group level.
    """
    return grps.data_map.loc['test']

def test(tst, tst_tags, tag_headers, tst_exp_path, model_args, prebuilt=None):
    metrics = {}
    # Change working directory and save old working directory 
    working_dir = os.getcwd()
    os.chdir(tst_exp_path)
    # Initialize logger
    og_logger = logger.Logger.CURRENT
    logger.configure('logs',  ['log'], log_prefix='test-')

    ####################################################################################
    # Instantiate config and extract model wrapper
    ####################################################################################
    logger.info("Instantiating model config...")
    trn_exp_path = join(os.path.dirname(tst_exp_path), ds.TRN_DIR)
    train.set_default_model_resolvers(tst=tst, trn_exp_path=trn_exp_path, tst_exp_path=tst_exp_path)
    model_args = train.instantiate_model_config(args_class=TestArgs, 
                                                model_args=model_args, 
                                                prebuilt=prebuilt)
    logger.info("SUCCESS: Model config instantiated")

    model_wrapper = model_args['model_wrapper']
    
    ####################################################################################
    # Iterate over different test sets and save results 
    ####################################################################################
    logger.info("Testing model...")
    test_timer = utils.Timer()
    test_timer.start()
    for t, tag in zip(tst, tst_tags):
        tag_label = '-'.join([str(t) for t in tag])
        logger.log("{:=^50}".format(tag_label))

        # if scaler:
        #     t.data = scaler(t.data)
        
        tst_set = (t.data, t.labels)
        dataset = model_wrapper.dataset(tst_set=tst_set, **model_args['dataset'])
        
        tst_preds = model_wrapper.predict(dataset.tst_set, **model_args['predict'])
        metrics[tag] = train.evaluate_metrics(t.labels, tst_preds,  **model_args['evaluate_metrics'])
        
        # TODO: Fix save preds
        # train.save_preds(file_name=ds.TST_PRED_FILE.format(tag_label),
        #                 preds=tst_preds,
        #                 labels=np.argmax(t.labels, axis=1).reshape(-1, 1))

        # TODO: Move to model wrapper 
        # if model_args['save_layer_outputs']:
        #     save_layer_outputs(file_name=ds.LAYER_OUTPUTS_FILE.format(tag_label),
        #                        model_wrapper=model_wrapper, 
        #                        data=t.data,
        #                        labels=t.labels,
        #                        batch_size=dataset.batch_size,
        #                        inputs=inputs,
        #                        **model_args['save_layer_outputs'])
    test_time = test_timer.stop()
    logger.info(f"SUCCESS: Testing complete. Time to complete {test_time}")
    
    # Save metrics
    logger.info("Saving metrics...")
    metrics = train.save_metrics(metrics=metrics, multi_idx_names=tag_headers)
    # metrics = train.save_metrics(metrics=metrics, multi_idx_names=tst_grp.index.names)
    logger.info("SUCCESS: Metric saved")
    
    logger.info("Saving metrics summary...")
    save_metric_summary(metrics=metrics, **model_args['save_metric_summary'])
    logger.info("SUCCESS: Metric summary saved")
    
    logger.Logger.CURRENT = og_logger 
    os.chdir(working_dir)

# Default metrics summaries for default metrics (see train.add_default_metrics() function).
METRICS_SUMMARY_DEFAULTS = dict(
    avg_pos_prob='mean',
    avg_neg_prob='mean',
    tn='sum',
    fp='sum',
    fn='sum',
    tp='sum',
    tpr='mean',
    tnr='mean',
    ppv='mean',
)

def save_metric_summary(metrics,
                        metrics_summary_overrides=dict(),
                        metrics_summary_default='mean'):
    """ Save metric summaries for each sub-experiment.
    
        If you do not know the name of a metric check any metric.csv file and to find
        corresponding metric names. Custom metric names can be given by passing the 
        'name' parameter to the corresponding tf.keras.metrics class.
    
        Args:
            metrics (dict): Metrics to be saved where the key corresponds to the metric
                name and the value corresponds to the metric value.
                
            metrics_summary_overrides (dict): Overrides default summary metric for default
                metrics and adds summary metrics for non-default metrics. See 
                METRICS_SUMMARY_DEFAULTS for each deafult metric summary.
            
            metrics_summary_default (dict): If no metric summary is given for non-default 
                metrics then this default metric summary is used.
    """
    metrics_summary_overrides = {**METRICS_SUMMARY_DEFAULTS, **metrics_summary_overrides}
    
    results_dir = ds.RESULTS_DIR
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    sub_exps = metrics.groupby(level=['dataset', 'subject'])
    metrics_summary = []
    metric_columns = ['sub-exp'] 
    
    for se_name, se_data in sub_exps:
        se_metrics_summary = ['-'.join([str(n) for n in se_name])]
        se_metric_columns = []
        
        for (metric_name, metric_data) in se_data.items():               
            if metric_name in metrics_summary_overrides:
                summary_metrics = metrics_summary_overrides[metric_name]
            else:
                summary_metrics = metrics_summary_default
                
            if not isinstance(summary_metrics, list):
                summary_metrics = [summary_metrics]
            
            # Compute summaries for each unique metric
            for sm in summary_metrics:
                if not hasattr(metric_data, sm):
                    err = "The summary metric {} was not found for data stored as type {}"
                    raise ValueError(err.format(sm, type(metric_data)))
                
                summary = getattr(metric_data, sm)
                se_metrics_summary.append(summary())

                sm_name = "{}_{}".format(metric_name, sm)
                if sm_name not in metric_columns:
                    metric_columns += [sm_name]
                       
        metrics_summary.append(se_metrics_summary)
    
    metrics_summary = pd.DataFrame(metrics_summary, columns=metric_columns)
    metrics_summary_file = join(results_dir, ds.METRICS_SUMMARY_FILE)
    metrics_summary.to_csv(metrics_summary_file, mode='w', header=True, index=False)

    return metrics_summary

# TODO: Move to model wrapper 
# TODO: This is slow
def save_layer_outputs(file_name, model_wrapper, data, labels, batch_size, inputs, use_class=None):
    labels = np.argmax(labels, axis=1) if labels.shape[-1] > 1 else labels
 
    if use_class is not None:
        if not isinstance(use_class, (list, tuple)):
            use_class = list(use_class)
        class_idx = np.hstack([np.where(labels)[0] for idx in use_class])
        data = data[class_idx]
        labels = labels[class_idx]
        
    dataset = train.build_tf_dataset(data=data,
                                     labels=labels,
                                     batch_size=batch_size, 
                                     shuffle=False,
                                     buffer_size=None)
    
    layer_outputs = model_wrapper.get_layer_outputs(dataset, inputs)
    layer_outputs = [outputs.astype(np.float16) for outputs in layer_outputs]
    custom_names, class_names = model_wrapper.get_layer_names()

    dir_path = ds.LAYER_OUTPUTS_DIR
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = join(dir_path, file_name)

    # with h5py.File(file_path+'.hdf5', 'w') as f:
    #     for i, output in enumerate(layer_outputs):
    #         dset = f.create_dataset(str(i), 
    #                                 data=output, 
    #                                 compression='lzf',
    #                                 chunks=output.shape)
    save_info = {
            "outputs": layer_outputs, 
            "custom_layer_names": custom_names, 
            "layer_class_names": class_names,
            "labels": labels
    }
    with open(file_path, 'wb') as f:
        dill.dump(save_info, f)
    # compress.zstd_compress(file_path)
    

@dataclass(order=True)
class _TestRequiredArgs():
    """
        model_wrapper (dict): 
            
    """
    model_wrapper: dict = None
    
@dataclass(order=True)
class _TestDefaultArgs():
    """ Dataclass that stores variables that will be used by the run_test.py script.
    
        Attributes:
        
            predict (dict): Kwargs for the model wrappers test() method
            
            mertics (list):  List of strings which correspond to metrics from 
                tf.keras.metrics or dbci.models.metrics.
            
            save_layer_outputs (dict | None): Dictionary containing keywords for storing layer
                outputs will be stored. Pass None to disable this feature. 

            save_metrics_summary (dict): Kwargs corresponding to test.save_metric_summary()
                function for summarizing the computed metrics.
                
             seed (int): Seed to use for any random processes that occur (e.g., during mutation)
            
    """
    predict: dict = field(default_factory=dict)
    save_layer_outputs: bool = False
    save_metric_summary: dict = field(default_factory=dict)
    seed: int = np.random.randint(99999)
    preinstantiate: dict = field(default_factory=dict)
    dataset: dict = field(default_factory=dict)
    evaluate_metrics: dict = field(default_factory=dict)

    def __getitem__(self, item):
        return getattr(self, item)
    
@dataclass(order=True)
class TestArgs(_TestDefaultArgs, _TestRequiredArgs):
    """ Class which combines both _TestDefaultArgs and _TestRequiredArgs"""
    
    def __getitem__(self, item):
        return getattr(self, item)
