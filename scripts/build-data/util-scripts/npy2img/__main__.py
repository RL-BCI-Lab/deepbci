import argparse
import os
from pdb import set_trace

import deepbci.utils.utils as utils 
from deepbci.data.scripts.npy2img.npy2img import npy2img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--task", help="Task name", required=True)
    parser.add_argument("-s", "--subject", help="Subject number (Ex: 01 or 10)", 
                        required=True)
    parser.add_argument("-tl", "--trial", help="Trial number", type=int, 
                        required=True)
    parser.add_argument("-e", "--extension", help="Image format", required=True)
    parser.add_argument("-u", "--upto", help="Loops up to a given trial", 
                        type=int)
    parser.add_argument("-r", "--rate", help="Rate at which images were recorded", 
                        default='1', type=int)
    parser.add_argument("-rnd", "--round", default='1', type=int,
                        help="Round to nearest decimal place when recreating frame names")
    parser.add_argument("-d", "--delete", help="Remove .npy file", action='store_true')
    parser.add_argument("-n", "--file-name", help="Compressed file name", 
                        type=str, default='state-images')

    args = parser.parse_args()
    if args.upto is None:
        args.upto = args.trial
    print('t:{} s:{} tl:{} e:{} u:{} r:{} rnd:{} d:{}, n:{}'.format(
        args.task, args.subject, args.trial, args.extension, args.upto, 
        args.rate, args.round, args.delete, args.file_name))
    
    for tl in range(args.trial, args.upto+1):
        print('Trial:', tl)
        # Creates path to <oa, obs>/S<number>/task-<number>/states
        file_path = os.path.join(
            args.task, 
            "S{}".format(args.subject),
            "trial-{}".format(tl), 
            "states")

        asb_folder_path = utils.path_to(utils.get_module_root(), file_path)
        save_loc = os.path.join(asb_folder_path, args.file_name + '.npy') 
        npy2img(folder=asb_folder_path, save_path=save_loc, 
                ext=args.extension, rate=args.rate, dec=args.round, 
                delete=args.delete)
