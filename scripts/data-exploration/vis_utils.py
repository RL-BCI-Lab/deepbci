import itertools
from copy import deepcopy
from pdb import set_trace

import pandas as pd
import numpy as np
from hydra.utils import instantiate

from deepbci.data_utils.data import Groups, run_group_mutators

def clean_config(cfg, keep_keys):
    keys = list(cfg.keys())
    [cfg.pop(k) for k in keys if k not in keep_keys]

def instantiate_and_mutate(data_cfg):
    built_data_cfg = instantiate(data_cfg, _convert_='all')

    grps = built_data_cfg.pop('groups')
    mutate = built_data_cfg.pop('mutate', None)

    if mutate is not None:
        run_group_mutators(grps, mutate)
                            
    return grps

def get_drop_logs(grps):
    drops, names = [], []
    for n, df in grps.data_map.groupby(grps._levels):
        raw = df[0].data
        dropped = len([t for t in raw.drop_log if len(t) != 0])
        total = len(raw.drop_log)
        percent_dropped = dropped / total
        names.append(n)
        info = (percent_dropped, dropped, total)
        drops.append(info)
    indexes = pd.MultiIndex.from_tuples(names, names=grps._levels)
    return pd.DataFrame(drops, columns=['percent', 'dropped', 'total'], index=indexes)


class Tunable():
    def __init__(self, values, name):
        self._check_values_type(values)
        self.values = values
        self.name = name
    
    def __iter__(self):
        for v in self.values:
            yield v

    def _check_values_type(self, values):
        if not isinstance(values, (list, tuple, np.ndarray)):
            err = f"The 'value' argument for tunable much be of type list, tuple, or np.ndarray: " \
                    f"received {type(values)}"
            raise TypeError(err)

        
    @staticmethod
    def tunable_constructor(loader, node):
        
        def select_constructor(loader, node, deep=False):
            classname = node.__class__.__name__
            if classname == "SequenceNode":
                return loader.construct_sequence(node, deep=deep)
            elif classname == "MappingNode":
                return loader.construct_mapping(node, deep=deep)
            else:
                return loader.construct_scalar(node)
            
        args = select_constructor(loader, node, deep=True)
        if isinstance(args, list):
            return Tunable(*args)
        elif isinstance(args, dict):
            return Tunable(**args)
        else:
            err = "Tunable can only be created used a dict or list"
            raise TypeError(err)
        
class CVConfigManager():
 
    def __init__(self, configs):
        self.configs = configs
        self.tunables = self._find_tunables()
        self.has_tunables = False if len(self.tunables) == 0 else True
        # if singles:
        #     self.combos = self._generate_singles()
        # else:
        self.combos = self._generate_combos()
    
    def __getitem__(self, item):
        return self.configs[item]
    
    def __iter__(self):
        if self.combos is not None:
            for c in self.combos:
                values_updated, configs = self._update_configs(c)
                yield values_updated, configs
        else:
            yield None, self.copy_configs()

    def _update_configs(self, combo):
        configs = self.copy_configs()
        values_updated = []
        for c, v in combo.items():
            update_mapping = (v['name'], v['value'])
            values_updated.append(update_mapping)
            
            index, path = c[0], c[1:]
            self._update_value(configs[index], path, v)
            
        return values_updated, configs
            
    def copy_configs(self):
        return deepcopy(self.configs)
    
    def _update_value(self, config, path, value):
        curr_loc = config
        last_idx = path[-1]
        
        for idx in path[:-1]:
            curr_loc = curr_loc[idx]
        curr_loc[last_idx] = value['value']
    
    # def _generate_singles(self):
    #     tunable_combos = []
    #     for key, tunable in self.tunables.items():
    #         value_info = []
    #         for value in tunable.values:
    #             info_dict = dict(
    #                 name=tunable.name,
    #                 value=value
    #             )
                
    #             value_info.append(info_dict)
    #         keys = [key]*len(value_info)
    #         set_trace()
    #         combo = dict(zip(keys, value_info))
    #         tunable_combos.append(combo)
        
    #     return tunable_combos
    
    def _generate_combos(self):
        if len(self.tunables) == 0:
            return None
        
        keys = list(self.tunables)
        tunable_combos = []
        tunables =  [*self.tunables.values()]
        
        for values in itertools.product(*tunables):
            value_info = []
            for t, v in zip(tunables, values):
                if v not in t.values:
                    err = 'Value {v} was not found for current Tunable: {t.values}!'
                    raise ValueError(err)
                
                info_dict = dict(
                    name=t.name,
                    value=v
                )
                value_info.append(info_dict)
            combo = dict(zip(keys, value_info))
            tunable_combos.append(combo)
 
        return tunable_combos

    def _find_tunables(self):
        tunables = {}
        for k, cfg in self.configs.items():
            tunables_found = self._tunable_search(cfg, index=k)
            tunables = {**tunables, **tunables_found}
        return tunables

    def _tunable_search(self, iterable, index):
        tunables = {}
        path = (index,)
        def find(v, path=(), found={}):
            
            def check(v, curr_path):
                if isinstance(v, Tunable):
                    found[curr_path] = v
                else:
                    find(v, curr_path, found)
                    
            def loop_dict(dict_, path):
                for k, v in dict_.items():
                    curr_path = path + (k,)
                    check(v, curr_path)
                  
                
            def loop_list(list_, path):
                for i, v in enumerate(list_):
                    curr_path = path + (i,)
                    check(v, curr_path)
            
            if isinstance(v, dict):
                loop_dict(v, path)
            elif isinstance(v, (list, tuple)):
                loop_list(v, path)

        find(iterable, path, tunables)
        return tunables