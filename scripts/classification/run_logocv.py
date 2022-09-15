import argparse
import os
from dataclasses import asdict
from os.path import join
from pdb import set_trace

import deepbci.utils.utils as utils

import directory_structure as ds
from logocv import KNestedLOGOCV, LOGOCVArgs, Tunable

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cross-validate model")

    parser.add_argument('--model-cfg', type=str, metavar='MODELCONFIG', required=True,
                        help="path to a model config within the classification/ dir ")
    parser.add_argument('--data-cfg', type=str, metavar='DATACONFIG', default=None,
                        help="path to a data config within the classification/ dir ")
    parser.add_argument('--exp-dir', type=str, metavar='EXPDIR', 
                        help="name of directory to be created within exps/ to store experiment output")

    # Get yaml loader and add new resolver for hyper-parameter tuning
    yaml_loader = utils.get_yaml_loader()
    yaml_loader.add_constructor(u'!Tunable', Tunable.tunable_constructor)
    
    # Load data and model configs
    cmd_args = parser.parse_args()

    # Generate classification/ directory structure
    ds.generate_directory_structure()
    
    # Optional load data config IF --load-cache is used.
    # If no --load-cache is used then data config file name is needed.
    if cmd_args.data_cfg is not None:
        data_cfg_path = join(ds.ROOT_DIR, cmd_args.data_cfg) 
        cmd_args.data_cfg = utils.load_yaml(data_cfg_path, loader=yaml_loader)

    # Load model (required)
    model_cfg_path = join(ds.ROOT_DIR, cmd_args.model_cfg)
    
    model_cfg = utils.load_yaml(model_cfg_path, loader=yaml_loader)
    cmd_args.model_cfg = asdict(
        LOGOCVArgs(**model_cfg)
    )

    cv = KNestedLOGOCV(
        data_cfg=cmd_args.data_cfg, 
        model_args=cmd_args.model_cfg,
        exp_dir=cmd_args.exp_dir
    )
    cv()
    