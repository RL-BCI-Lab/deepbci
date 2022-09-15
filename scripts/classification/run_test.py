import argparse
import os
from dataclasses import asdict
from os.path import join
from pdb import set_trace

import deepbci.utils.utils as utils

import directory_structure as ds 
from test import main, TestArgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test model")

    parser.add_argument('--exp-dir', type=str, metavar='EXPERIMENTDIR', required=True,
                        help="name of experiment directory to load within scripts/classification/exp/ dir")
    parser.add_argument('--model-cfg', type=str, metavar='MODElCONFIG', required=True,
                        help="path to a model config within the classification/ dir")
    parser.add_argument('--data-cfg', type=str, metavar='DATACONFIG', default=None,
                        help="path to a data config within the classification/ dir")
    parser.add_argument('--cache', type=str, nargs='?', const='trn-cache',
                            metavar='CACHE', help="name of dir which will store cached data within cache/")
    parser.add_argument('--load-cache', type=str, nargs='?', const='trn-cache', 
                        metavar='LOADCACHE', help="name of cache data file to load within cache/")
    # Load data and model configs
    cmd_args = parser.parse_args()
    
    # Generate classification/ directory structure
    ds.generate_directory_structure()

    # Optional load data config IF --load-cache is used.
    # If no --load-cache is used then data config file name is needed.
    if cmd_args.data_cfg is not None:
        data_cfg_path = join(ds.ROOT_DIR, cmd_args.data_cfg) 
        cmd_args.data_cfg = utils.load_yaml(data_cfg_path)

    # Load model (required)
    model_cfg_path = join(ds.ROOT_DIR, cmd_args.model_cfg)
    cmd_args.model_cfg = asdict(
        TestArgs(**utils.load_yaml(model_cfg_path))
    )

    # Crate, cache dir and build cache paths
    cache_dir_path = join(ds.ROOT_DIR, ds.CACHE_DIR)

    if cmd_args.load_cache is not None:
        cmd_args.load_cache = join(cache_dir_path, cmd_args.load_cache)
        if not os.path.exists(cmd_args.load_cache):
            err = "Cached file {} was not found within the cache/ directory"
            raise ValueError(err.format(os.path.basename(cmd_args.load_cache)))
        
    if cmd_args.cache is not None:
        cmd_args.cache = join(cache_dir_path, cmd_args.cache)

    if cmd_args.exp_dir[-1] == os.path.sep:
        cmd_args.exp_dir = cmd_args.exp_dir[:-1]

    # Check if exp_path even exists
    if ds.EXPS_DIR in cmd_args.exp_dir:
         exp_path = join(ds.ROOT_DIR, cmd_args.exp_dir)
    else:
        exp_path = join(ds.ROOT_DIR, ds.EXPS_DIR, cmd_args.exp_dir)
    if not os.path.exists(exp_path):
        err = "Passed experiment directory {} does not exist within {}"
        raise IOError(err.format(os.path.basename(cmd_args.exp_dir), 
                                 join(ds.ROOT_DIR, ds.EXPS_DIR)))
    
    main(
        model_args=cmd_args.model_cfg,
        data_cfg=cmd_args.data_cfg, 
        exp_dir=cmd_args.exp_dir, 
        load_cache= cmd_args.load_cache, 
        cache= cmd_args.cache
    )