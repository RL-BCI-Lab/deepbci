import argparse
import os
from os.path import join, dirname
from pdb import set_trace

from deepbci.utils.loading import load_config, path_to
from .agent_controller import AgentController
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,  required=True, help="Config name")
    args = parser.parse_args()

    config = '{}.yml'.format(args.config)
    kwargs = load_config(join(dirname(__file__), "configs", config))

    agent = AgentController(**kwargs)
    agent.run()