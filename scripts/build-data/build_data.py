"""
    Script for building deepbci datasets. Works by combining task and EEG related 
    information to create the forward facing deepbci datasets. This data is the
    "true" raw data that needs to be properly combined to create useable datasets.
    
    TODO: Add proper logging of ExperimentDataBuilder
"""

import argparse
from pdb import set_trace
from os.path import join

from hydra.utils import instantiate

import deepbci.utils.utils as utils
import deepbci.utils.logger as logger
from deepbci.data_utils.build_data.experiment_data_builder import ExperimentDataBuilder

def clean_config(config, keep_keys):
    keys = list(config.keys())
    [config.pop(k) for k in keys if k not in keep_keys]
    return config

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, metavar='CONFIG', required=True,
                        help="path to a config for building data")
parser.add_argument('--dry-run', action='store_true', 
                    help="dry run script without saving cleaned files")
parser.add_argument('--verbose', action='store_true', 
                    help="build data with verbose logging")
args = parser.parse_args()

config = utils.load_yaml(args.config)
inst_config = clean_config(
    config=instantiate(config,  _convert_='all'),
    keep_keys=['datasets', 'file_name', 'build_kwargs']
)
builder = ExperimentDataBuilder(verbose=args.verbose)

builder.build_to_disk(dry_run=args.dry_run, **inst_config,)
