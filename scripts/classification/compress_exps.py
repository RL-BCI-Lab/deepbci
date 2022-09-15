# import required modules
import os
import zipfile
import tarfile
import argparse
from pathlib import Path
from pdb import set_trace

from tqdm import tqdm

import directory_structure as ds
 
# Declare the function to return all file paths of the particular directory
def get_files(dirname, exclude=None):
    exclude = [] if exclude is None else exclude
    
    # setup file paths variable
    file_paths = []
    # Read all directory, subdirectories and file lists
    for root, dirs, files in os.walk(dirname):

        dirs[:] = [d for d in dirs if d not in exclude]
        for filename in files:
            if filename in exclude:
                continue
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)

    return file_paths
 
def generate_name(name):
    return name.name[:-1] if str(name).endswith(os.path.sep) else name.name
 
def main(dirname, compress_name=None, exclude=None, dryrun=False):
    dirname = Path(dirname)
    
    og_dir = os.getcwd()
    os.chdir(dirname.parent)
    
    dirname = Path(dirname.name)

    compress_name = generate_name(dirname) if compress_name is None else compress_name
    compress_path = dirname.parent / Path(compress_name)
    if compress_path.suffix != '.tar.gz':
        compress_path = compress_path.with_suffix('.tar.gz')

    print(f"Excluding the following files/dirs: {exclude}")
    # Call the function to retrieve all files and folders of the assigned directory
    file_paths = get_files(dirname, exclude=exclude)
        
    if dryrun:
        print('The following files will be compressed:')
        for f in file_paths:
            print(f)
        print(f"The compressed file will be save to '{os.path.join(os.getcwd(), compress_path)}'")
    else:
        with tarfile.open(compress_path, "w:gz") as tar:
            # writing each file one by one
            for f in tqdm(file_paths, desc="Compressing files..."):
                tar.add(f)
        print(f"The directory '{dirname}' was successfully compressed to {os.path.join(os.getcwd(), compress_path)}")
        print(f"Compression size ~{os.path.getsize(compress_path) >> 20} MB")
        
    os.chdir(og_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument('-d', '--dir', type=str, metavar='DIRECTORY', required=True,
                        help="Path to directory which will be compressed")
    parser.add_argument('-n', '--name', type=str, metavar='FILENAME', required=False,
                        help="Name of compressed file")
    parser.add_argument('--dry-run', dest='dryrun', action='store_true', help='Perform a dry run')
    parser.add_argument('-e', '--exclude', metavar='EXCLUDE',  nargs='+',
                        help="Name of subdirectories or files to exclude")
    # Load data and model configs
    cmd_args = parser.parse_args()
 
    main(dirname=cmd_args.dir, 
         compress_name=cmd_args.name, 
         exclude=cmd_args.exclude, 
         dryrun=cmd_args.dryrun)