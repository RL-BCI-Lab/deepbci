import argparse
from pdb import set_trace
from os.path import join

from hydra.utils import instantiate

import deepbci.utils.utils as utils
from deepbci.data_utils.build_data.exp_data_builder import ExpDataBuilder

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true', 
                    help="dry run script without saving cleaned files")
parser.add_argument('--verbose', action='store_true', 
                    help="build data with verbose logging")
args = parser.parse_args()

config_name = 'data-settings.yaml'
config_rel = join('configs', config_name)
config = utils.load_yaml(config_rel)

inst_config = instantiate(config,  _convert_='all')
builder = ExpDataBuilder()
builder.build_to_disk(inst_config['datasets'], inst_config['build_kwargs'], dry_run=args.dry_run, verbose=args.verbose)
