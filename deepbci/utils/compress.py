import subprocess
import os

def zstd_compress(file_path, clean=False):
    """ Compress a given file with Facebook's zstd compression algorithm

        Note:
            Utilizes zstd via Python commandline tools.

        Args:
            save_path (str): Absolute path to file you wish to compress.

            clean (bool): If true remove uncompressed file. 
    """
    print("Attempting to compress {}...".format(file_path))
    code = subprocess.run(["zstd", "-zf", file_path]).returncode
    if code != 0:
        print("Failed zstd compression...")
        print("Checking for zstd..")
        if subprocess.run("zstd").returncode != 0:
            print("zstd found...")
            raise Exception("zstd failed to compress, check stdout for error...")
        else:
            raise Exception("No zstd found!")
    print("Compression was successful!")
    if clean: 
        print("Attemtping to remove uncompressed file...")
        os.remove(file_path)
        print("Uncompressed file successfully removed!")
    

def zstd_decompress(zstd_path):
    if os.path.exists(zstd_path):
        print("Attempting to decompress {}...".format(zstd_path))
        code = subprocess.run(["zstd", "-df", zstd_path]).returncode
        if code != 0:
            print("Failed zstd decompression...")
            print("Checking for zstd..")
            if subprocess.run("zstd").returncode != 0:
                print("zstd found...")
                raise Exception("zstd failed to decompress, check stdout for error...")
            else:
                raise Exception("No zstd found!")
    else:
        raise Exception("No .zst file detected...")
    print("Decompression was successful!")