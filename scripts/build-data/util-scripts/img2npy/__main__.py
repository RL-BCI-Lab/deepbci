import argparse
import os
from pdb import set_trace

from utils.loading import path_to
from .img2npy import img2npy

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
    parser.add_argument("-d", "--delete", help="Delete images after storing ", 
                        action='store_true')
    parser.add_argument("-r", "--resize", help="Resize images, input width height", 
                        nargs=2, type=int)
    parser.add_argument("-g", "--grayscale", help="Convert images to grayscale", 
                        action='store_true')
    parser.add_argument("-n", "--file-name", help="Compressed file name", 
                        type=str, default='state-images')

    args = parser.parse_args()
    if args.upto is None:
        args.upto = args.trial
        
    print('t:{} s:{} tl:{} e:{} u:{} d:{} r:{} g:{} n:{}'.format(
        args.task, args.subject, args.trial, args.extension, args.upto,
        args.delete, args.resize, args.grayscale, args.file_name))

    for tl in range(args.trial, args.upto+1):
        print('Trial:', tl)
        # Creates path to <oa, obs>/S<number>/task-<number>/states
        file_path = os.path.join(
            args.task, 
            "S{}".format(args.subject),
            "trial-{}".format(tl), 
            "states")
        abs_folder_path = path_to(os.getcwd(), file_path)
        save_path = os.path.join(abs_folder_path, args.file_name + '.npy') 
        img2npy(folder=abs_folder_path, save_path=save_path,    
                ext=args.extension, delete_imgs=args.delete,
                grayscale=args.grayscale, resize=args.resize)
    