import argparse
import os
from dataclasses import dataclass, asdict, field
from os.path import join
from pdb import set_trace

import numpy as np

import deepbci.utils.utils as utils

import directory_structure as ds  
from train import main

import subprocess
import os
import logging
import pickle
from pdb import set_trace
from typing import List, Sequence, Union
from pathlib import Path
from dataclasses import dataclass

import omegaconf
import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, to_absolute_path, get_method

log = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def print_config(
    cfg: DictConfig,
    style: str = 'dim',
    file: str = None
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

        Reference: 
            https://github.com/ashleve/lightning-hydra-template
        
        Args:
            config (DictConfig): Configuration composed by Hydra.
            
            fields (Sequence[str], optional): Determines which main fields from config will
                be printed and in what order.
                
            resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """
    def build_branch(tree, field, style, guide_style):
        if isinstance(field, DictConfig):
            for key, value in field.items():
                if isinstance(value, DictConfig):
                    branch = tree.add(key, style=style, guide_style=style)
                    build_branch(branch, value, style, guide_style)
                else:
                    tree.add(rich.syntax.Syntax(f"{key}: {value}", "yaml"))
                    
    tree = rich.tree.Tree(":gear: CONFIG", style=style, guide_style=style)
    build_branch(tree, cfg, style, style)

    if file is not None:
        with open(file, 'a') as f:
            rich.print(tree, file=f)
    else:
        rich.print(tree)

def set_default_resolvers():
    OmegaConf.register_new_resolver('get', get_method, replace=True)
    OmegaConf.register_new_resolver('original_cwd', get_original_cwd, replace=True)
    OmegaConf.register_new_resolver('abs_path', to_absolute_path, replace=True)
    OmegaConf.register_new_resolver('join', join, replace=True)
    
def hydra_main(func, cfg_path: str, cfg_name: str = 'config'):
    cfg_path = Path(cfg_path) if not isinstance(cfg_path, Path) else cfg_path
    cfg_name = Path(cfg_name) if not isinstance(cfg_name, Path) else cfg_name
    
    if not cfg_path.is_absolute():
        err = f"cfg_path received {cfg_path} which is not an absolute path." \
              "An absolute path is required in order to find the correct config."
        raise ValueError(err)
    # Override hydra automatic detection of a module it will try to use configs
    # located in a Python module even if they don't exist!
    os.environ['HYDRA_MAIN_MODULE'] = '__main__'
    
    @hydra.main(config_path=str(cfg_path), config_name=str(cfg_name))
    def actual_hydra_main(cfg: DictConfig) -> None:
        # Remove any extra parent dirs as Hydra nests the actual config within 
        # any parent dirs specified in the config_name or add "# @package _global_" to config.
        for parent in cfg_name.parent.parts:
            if parent in cfg:
                cfg = cfg[parent]
        set_trace()
        
        # log.debug("Instantiating objects in config...")
        # inst_cfg = hydra.utils.instantiate(cfg)

        if cfg.get('print_config'):
            print_config(cfg, **cfg.print_config)
        set_trace()    
        
        func()
    
    # Set default resolvers
    set_default_resolvers()
    
    # Run actual training/testing using Hydra
    actual_hydra_main()
   
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Train model")

    # parser.add_argument('--cache', type=str, nargs='?', const='trn-cache',
    #                     metavar='CACHE', help="name of dir which will store cached data within scripts/classification/cache/")
    # parser.add_argument('--load-cache', type=str, nargs='?', const='trn-cache', 
    #                     metavar='LOADCACHE', help="name of cache data file to load within scripts/classification/cache/")
    # # Load data and model configs
    # cmd_args = parser.parse_args()
    
    
    path = Path(os.getcwd())/'hydra_configs'
    hydra_main(func=main, cfg_path=str(path))

    # Generate classification/ directory structure
    # ds.generate_directory_structure()

    # Optional load data config IF --load-cache is used.
    # If --load-cache is not used then data config file name is needed.
    # if cmd_args.data_cfg is not None:
    #     data_cfg_path = join(ds.ROOT_DIR, cmd_args.data_cfg) 
    #     cmd_args.data_cfg = utils.load_yaml(data_cfg_path)

    # # Load model (required)
    # model_cfg_path = join(ds.ROOT_DIR, cmd_args.model_cfg)
    # cmd_args.model_cfg = TrainArgs(**utils.load_yaml(model_cfg_path))

    # # Crate, cache dir and build cache paths
    # cache_dir_path = join(ds.ROOT_DIR, ds.CACHE_DIR)

    # if cmd_args.load_cache is not None:
    #     cmd_args.load_cache = join(cache_dir_path, cmd_args.load_cache)
    #     if not os.path.exists(cmd_args.load_cache):
    #         err = "Cached file {} was not found within the cache/ directory"
    #         raise ValueError(err.format(os.path.basename(cmd_args.load_cache)))
        
    # if cmd_args.cache is not None:
    #     cmd_args.cache = join(cache_dir_path, cmd_args.cache)
        
    
    set_trace()
    main(args=cmd_args.model_cfg,
         data_cfg=cmd_args.data_cfg, 
         exp_dir=cmd_args.exp_dir, 
         load_cache=cmd_args.load_cache, 
         cache=cmd_args.cache)
         