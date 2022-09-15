"""
Runs multiple experiments based on a experiment config template (templating using jinja2).
The corresponding definition config file is needed to fill in the experiment config template.
All experiment and definition configs should be referenced using a local path and must 
be located withing the /deep-bci/experiments/classification/ directory.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
import traceback
import os
import argparse
import shutil
import multiprocessing as mp
from functools import partial 
from dataclasses import asdict
from os.path import join 
from pdb import set_trace

import yaml
import jinja2
import ray
# Mutes tensorflows excessive messaging which is made worse
# when multiple GPUs are in use.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from deepbci.utils import logger
from deepbci.utils import utils

import train
import test
import directory_structure as ds
from logocv import KNestedLOGOCV, LOGOCVArgs, Tunable
from utils import generate_experiments_summary

def main(exp_cfg_path, def_cfg_path, method_type, **kwargs):
    # Generate directory structure
    ds.generate_directory_structure()
    
    yaml_loader = utils.get_yaml_loader()
    if method_type == 'logocv':
        yaml_loader.add_constructor(u'!Tunable', Tunable.tunable_constructor)
        
    def_cfg_path = join(ds.ROOT_DIR, def_cfg_path)
    def_cfg = utils.load_yaml(def_cfg_path, loader=yaml_loader)

    save_exp_configs(exp_cfg_path, def_cfg_path, parent_dir=def_cfg['parent_dir'])
    
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=ds.ROOT_DIR))
    exp_template = env.get_template(exp_cfg_path)
    exp_rendered = exp_template.render(def_cfg)
    exp_cfg = yaml.load(exp_rendered, Loader=yaml_loader)

    method_type = method_selector(method_type, **kwargs)
    method_type(exp_cfg)

def save_exp_configs(exp_cfg_path, def_cfg_path, parent_dir):
    cfg_dir = join(ds.EXPS_DIR, parent_dir, ds.CFG_DIR)
    
    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)
        
    if exp_cfg_path:
        exp_cfg_save_path = join(cfg_dir, ds.EXP_FILE)
        shutil.copyfile(exp_cfg_path, exp_cfg_save_path)
    
    if def_cfg_path:
        def_cfg_save_path = join(cfg_dir, ds.EXP_DEF_FILE)
        shutil.copyfile(def_cfg_path, def_cfg_save_path)
  
def method_selector(method_type, num_cpus, cpus_per_task, num_gpus=0, gpus_per_task=0):
    kwargs = dict(
        num_cpus=num_cpus, 
        cpus_per_task=cpus_per_task,
        num_gpus=num_gpus,
        gpus_per_task=gpus_per_task, 
    )
    
    if method_type == 'train_and_test':
        raise Exception("Not implemented yet")
        # kwargs['func'] = run_train_and_test
    elif method_type == "logocv":
        kwargs['func'] = run_logocv
    elif method_type == 'test':
         kwargs['func'] = ray_test
    else:
        err = "Invalid method_type {} was passed"
        raise TypeError(err.format(method_type))
    
    return partial(run_experiments, **kwargs)

def run_experiments(exp_cfg, func, num_cpus, cpus_per_task, num_gpus=0, gpus_per_task=0):
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
    func_exp = ray.remote(num_cpus=cpus_per_task, num_gpus=gpus_per_task)(func)
    
    exps = []
    exp_paths = []
    for cfg in exp_cfg['exps']:
        (exp_dir, kwargs), = cfg.items()
        # Build path to exp directory
        exp_path = join(ds.ROOT_DIR, ds.EXPS_DIR, exp_dir)
        exp_paths.append(exp_path)
        
        kwargs['exp_dir'] = exp_dir
        exps.append(func_exp.remote(**kwargs))
        
    ray_job_status(jobs=exps)
    output = [ray.get(e) for e in exps]
    ray.shutdown()
 
def ray_test(exp_dir, **kwargs):
    print("{:-^50s}".format(exp_dir))
    import numpy as np
    wait = np.random.randint(5, 10)
    
    print(f"Watinging: {wait} seconds")
    physical_gpus = tf.config.list_physical_devices('GPU')
    [tf.config.experimental.set_memory_growth(g, True) for g  in physical_gpus]
    print(f"GPUs configured: {tf.config.list_logical_devices('GPU')}")
    
    time.sleep(wait)

    return wait

# def run_train_and_test(exp_dir, **kwargs):
#     """ Executes training and testing"""
#     # Get data config for constructing data
#     data_cfg = kwargs.pop('trn-data-cfg')
#     # Get and load arguments for training/network
#     model_cfg = train.TrainArgs(**kwargs.pop('trn-args-cfg'))
#     train.main(model_args=model_cfg,
#                data_cfg=data_cfg, 
#                exp_dir=exp_dir)
    
#     data_cfg = kwargs.pop('tst-data-cfg')
#     args = test.TestArgs(**kwargs.pop('tst-args-cfg'))
#     test.main(model_args=model_cfg,
#               data_cfg=data_cfg, 
#               exp_dir=exp_dir)
    
def run_logocv(exp_dir, **kwargs):
    data_cfg = kwargs.pop('data-cfg')
    model_args = asdict(LOGOCVArgs(**kwargs.pop('model-cfg')))
    cv = KNestedLOGOCV(data_cfg=data_cfg,
                       model_args=model_args,
                       exp_dir=exp_dir)()


def ray_job_status(jobs, interval=10):
    import psutil
    import time
    import datetime
    start = time.time()
    # print(psutil.virtual_memory().total)
    print(f"Total jobs: {len(jobs)}")
    while True:
        ready, not_ready = ray.wait(jobs, timeout=5, num_returns=len(jobs))
        if len(not_ready) > 0:
            cur_time = datetime.timedelta(seconds=time.time() - start)
            print('Time: {} Jobs Finished: {}/{} Memory: {} GB'.format(str(cur_time), 
                                                                len(ready),
                                                                len(jobs),
                                                                psutil.virtual_memory()[2]))
            time.sleep(interval)
        else:
            break
    end_time = datetime.timedelta(seconds=time.time() - start)
    print(f"Total execution time: {end_time}")

# def run_multi_gpu_exps(exp_cfg, func, n_gpus):
#     """ Runs multiple basic training and then testing experiments """
#     exp_paths = []
#     jobs = []
#     results = []
#     log_results = lambda result: results.append(result)

#     # Build available queue which contains GPUs which are aviable for use
#     manager = mp.Manager()
#     avail_gpus = manager.Queue(maxsize=n_gpus)
#     for i in range(n_gpus):
#         avail_gpus.put(i)

#     pool = mp.Pool(n_gpus)

#     for cfg in exp_cfg['exps']:
#         (exp_dir, kwargs), = cfg.items()
#         # Build path to exp directory
#         exp_path = join(ds.ROOT_DIR, ds.EXPS_DIR, exp_dir)
#         exp_paths.append(exp_path)
#         # Add additional kwargs
#         kwargs['func'] = func
#         kwargs['avail_gpus'] = avail_gpus
#         kwargs['exp_dir'] = exp_dir
#         # Initialize processes
#         jobs.append(pool.apply_async(multi_gpu, 
#                                      kwds=kwargs, 
#                                      callback=log_results))
        
#     job_status(jobs)
#     pool.close()
#     pool.join()
#     print([j.successful() for j in jobs])
#     generate_experiments_summary(exp_paths)
#     # print("Results: {}".format(results))
    
# def multi_gpu(func, avail_gpus, *args, **kwargs):
#     # Wrap in try-catch so errors will actually get printed
#     try:
#         # Get gpu for this process
#         gpu = avail_gpus.get()
        
#         # Set GPU visible device and set memory growth
#         physical_gpus = tf.config.list_physical_devices('GPU')
#         tf.config.experimental.set_visible_devices(physical_gpus[gpu], 'GPU')
#         tf.config.experimental.set_memory_growth(physical_gpus[gpu], True)
#         print(f"configured: {gpu}")
#         # Run function given selected gpu
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         with tf.device(logical_gpus[0]):
#             results = func(*args, **kwargs)
            
#         # Add gpu back to being available 
#         avail_gpus.put(gpu)
#     except Exception as e:
#         raise Exception(e)
    
#     return results
    
# def run_exps(exp_cfg, func):
#     """ Runs multiple basic experiments with training and testing """
#     utils.tf_allow_growth()
#     exp_paths = []
#     for cfg in exp_cfg['exps']:
#         # Extract experiment local path and configs
#         (exp_dir, kwargs), = cfg.items()

#         exp_path = join(ds.ROOT_DIR, ds.EXPS_DIR, exp_dir)
#         exp_paths.append(exp_path)
        
#         func(exp_dir, **kwargs)
#     generate_experiments_summary(exp_paths)
    
# def job_status(jobs, interval=60):
#     import psutil
#     import time
#     import datetime
#     start = time.time()
#     while True:
#         check_jobs = [j.ready() for j in jobs]
#         if not all(check_jobs):
#             jobs_left = len(jobs) - sum(check_jobs)
#             cur_time = datetime.timedelta(seconds=time.time() - start)
#             print('Time: {} Tasks Left: {} Memory: {}'.format(str(cur_time), 
#                                                                 jobs_left,
#                                                                 psutil.virtual_memory()[2]))
#             time.sleep(interval)
#         else:
#             break
        

if __name__ == '__main__':
    # Arguments to be passed from command line
    parser = argparse.ArgumentParser(description="Running Multi-Experiments")
    parser.add_argument('--exp-cfg', type=str, required=True, metavar='EXPERIMENTCONFIG', 
                        help="path to a experiment config within the classification/ dir")
    parser.add_argument('--def-cfg', type=str, required=True, metavar='DEFINITIONCONFIG', 
                        help="path to a definition config within the classification/ dir")
    parser.add_argument('--method-type', type=str, default='train_and_test', metavar='METHODTYPE', 
                        help="type of method to use for training and testing")
    parser.add_argument('--cpus', type=int, metavar='CPU', 
                        help="Total number of CPUs avaliable")
    parser.add_argument('--cpus_per_task', type=int, default=1, metavar='PER_CPU', 
                        help="Number of CPUs used per job")
    parser.add_argument('--gpus', type=int, default=0, metavar='GPU', 
                        help="Number of GPUs avaliable")
    parser.add_argument('--gpus_per_task', type=int, default=0, metavar='PER_GPU', 
                        help="Number of GPUs used per job")
    cmd_args = parser.parse_args()
    
    main(
        exp_cfg_path=cmd_args.exp_cfg, 
        def_cfg_path=cmd_args.def_cfg, 
        method_type=cmd_args.method_type,
        num_cpus=cmd_args.cpus,
        cpus_per_task=cmd_args.cpus_per_task,
        num_gpus=cmd_args.gpus,
        gpus_per_task=cmd_args.gpus_per_task
    )