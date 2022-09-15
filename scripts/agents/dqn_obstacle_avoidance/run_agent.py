import argparse
import os
from pathlib import Path
from os.path import join
from pdb import set_trace

from deepbci.utils.utils import load_yaml, get_module_root
from agent_controller import AgentController

WORKING_DIR =  str(Path(__file__).parent.absolute())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,  required=True, 
                        help="Path to config within local directory")
    args = parser.parse_args()
    
    kwargs = load_yaml(join(WORKING_DIR, args.config))
    
    agent = AgentController(**kwargs)
    agent.run()
