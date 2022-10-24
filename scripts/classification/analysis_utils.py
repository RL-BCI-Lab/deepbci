import requests
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

class DownloadResults():
    def __init__(self, results_info, save_dir):
        self.save_dir = save_dir
        if not self.save_dir.exists():
            self.save_dir.mkdir()
        self.results_info = results_info
    
    def __call__(self):
        for result_name, info in self.results_info.items():
            directory_path = self._build_directory_path(info['directory'])
            tar_file_path = self._build_file_path(info['file'])
            
            if directory_path.exists():
                print(f"Skipping {result_name}, already exists")
                continue
            if tar_file_path.exists():
                self._decompress(tar_file_path)
            else:
                self._download_data(info['url'], tar_file_path)
                self._decompress(tar_file_path)
        print("Done")
    
    def _build_directory_path(self, directory):
        return self.save_dir / directory
    
    def _build_file_path(self, tar_file):
        return self.save_dir / tar_file
    
    def _decompress(self, tar_file_path):
        print(f"Extracting {tar_file_path.name}...")
        with tarfile.open(tar_file_path) as f:
            f.extractall(self.save_dir)
        if not tar_file_path.exists():
            raise FileNotFoundError(f"Results {tar_file_path} did not decompress.")
    
    def _download_data(self, url, tar_file_path):
        print(f"Downloading {tar_file_path.name}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(tar_file_path, 'wb') as f:
                f.write(response.raw.read())
        if not tar_file_path.exists():
            raise FileNotFoundError(f"Results {result_name} did not download.")

class DataFrameIndexWrapper:
    def __init__(self, df):
        self.df = df
        
    def __getitem__(self, index):
        index = tuple([[str(i)] if not isinstance(i, (list, tuple)) else list(map(str, i)) for i in index])
        return self.df.loc[index]
    
    def loc(self, *args, **kwargs):
        if len(args) == 0:
            args = self.df.columns
        index = [kwargs.get(idx, slice(None)) for idx in self.df.index.names]
        return self.df.loc[tuple(index), tuple(args)]
    
    def __repr__(self):
        return f"DataFrameIndexWrapper(shape={self.df.shape}"
    
    def _repr_html_(self):
        """Return a html representation for a particular DataFrame.

            Mainly for IPython notebook.
        """
        return self.df._repr_html_()
    
    def get_group_data(self, level, score):
        grps = self.df.groupby(level=level)
        dfs = []
        for name, df in grps:
            if isinstance(name, tuple):
                group_name = '-'.join(name)
            else:
                group_name = name
            grp_df = pd.DataFrame({
                'group': np.repeat([group_name], repeats=len(df)),
                'score': df[score].reset_index(drop=True), 
            })

            dfs.append(grp_df)
        dfs = pd.concat(dfs, axis=0)
        dfs.set_index('group', inplace=True)
        return dfs        
    
def debug_groupby(groups):
    for name, grp in groups:
        display(grp)
