import os
import gc
import itertools
from copy import copy, deepcopy
from os.path import join
from dataclasses import dataclass, asdict, field
from pdb import set_trace
from collections.abc import Iterable

import numpy as np
import pandas as pd

import deepbci as dbci
import deepbci.data_utils as dutils
import deepbci.utils.logger as logger
from deepbci.utils import utils

import directory_structure as ds
import train
import test
from train import _TrainRequiredArgs, _TrainDefaultArgs, TrainArgs
from test import _TestRequiredArgs, _TestDefaultArgs, TestArgs
from utils import generate_experiments_summary, set_all_seeds

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
    
class Folds():
    def __init__(self,):
        self.vld = []
        self.tst = None
        self.tst_combine = []
    
class CVConfigManager():
 
    def __init__(self, configs):
        self.configs = configs
        self.tunables = self._find_tunables()
        self.has_tunables = False if len(self.tunables) == 0 else True
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
 
class KNestedLOGOCV():
    def __init__(self, data_cfg, model_args, exp_dir):
        self._data_cfg = data_cfg
        self._model_args = model_args
        # 'groups' is excluded from using tunables
        self.cm = CVConfigManager(dict(mutate=data_cfg['mutate'], 
                                       model=model_args))
        
        self.exp_dir = exp_dir
        if self.exp_dir is None:
            self.exp_dir = ds.LOGO_EXP_DIR
        self.exp_path = train.build_exp_path(exp_dir)

        train.set_default_resolvers()

    def __call__(self, save_configs=True):
        # Change current working directory to experiment path
        og_path = self._chdir(self.exp_path)
        # Initialize logger
        self.logger = self._init_logger(self.exp_path)
        
        logger.info("Beginning LOGOCV...")
        timer = utils.Timer()
        timer.start()
        
        # Extract universal seed from model args
        seed = self.cm['model']['seed']
        self.rng = np.random.RandomState(seed)
        logger.info(f"Setting CV seed to {seed}", logger=self.logger)
        
        # Save configs
        train.save_configs(
            data_cfg=self._data_cfg , 
            model_args=self._model_args, 
            prefix="logocv"
        )
        
        # Pre-build data using groups config
        logger.info(f"Instantiating Groups object...", logger=self.logger)
        self.grps = train.instantiate_data_config(self._data_cfg['groups'], seed=seed)
        logger.info(f"SUCCESS: Groups object instantiated", logger=self.logger)
        
        # logger.info(f"Building Groups object instance...", logger=self.logger)
        # self.grps = train.build_groups(groups=groups_cfg, seed=seed)
        # logger.info(f"SUCCESS: Groups object instance built", logger=self.logger)

        # Generate data folds (excluding hyper-parameter tuning folds)
        logger.info(f"Generating folds...", logger=self.logger)
        folds = self.generate_repeated_folds(
            self.grps.data_map, 
            self.cm['model']['logo_level']
        )
        logger.info(f"SUCCESS: Total folds {len(folds)}", logger=self.logger)

        fold_paths = self.test_loop(folds)
        generate_experiments_summary(fold_paths)
    
    def _chdir(self, path):
        original_dir = os.getcwd()
        os.chdir(path)
        new_path = os.getcwd()
        return original_dir
    
    def _reset(self, path):
        # Revert logger and working directory back 
        logger.Logger.CURRENT = self.logger
        self._chdir(path)

    def _init_logger(self, path):
        logger.configure(join(path, 'log'),  ['log'], log_prefix='logo-')
        return logger.Logger.CURRENT
    
    def test_loop(self, folds):
        fold_paths = []
        for i, fold in enumerate(folds):
            tst_fold = fold.tst
            vld_fold = fold.vld
            
            # Generate directory for current test fold
            tst_fold_path = join(os.getcwd(), '-'.join(tst_fold[0][1:]))
            if not os.path.exists(tst_fold_path):
                os.makedirs(tst_fold_path)
            fold_paths.append(tst_fold_path)
            og_path = self._chdir(tst_fold_path)
            
            if self.cm.has_tunables:
                raise NotImplementedError("Hyper-parameter tuning not yet implemented")
                # self.valid_loop(fold)
                # mutate, model_args = self.cm.update_configs(self.best_combo)
                # self._train_and_test(tst_fold, mutate, model_args)
            else:
                cfgs = self.cm.copy_configs()
                fold_grps = self.grps.deepcopy()
                
                trn_grp = fold_grps.data_map.drop(tst_fold)
                tst_grp = fold_grps.data_map.loc[tst_fold].rename(index=dict(train='test'))
                fold_grps.data_map = pd.concat([trn_grp, tst_grp]).sort_index()
                self._log_fold_info(i+1, len(folds), tst_fold, fold_grps)
                self._train_and_test(fold_grps, cfgs['model'], cfgs['mutate'])
                
            self._reset(og_path)
            
        return fold_paths

    def _log_fold_info(self, iter, length, leave_out, grps):
        fold_title = "FOLD - {}/{} - Leave Out: {}".format(iter, length, leave_out)
        logger.info("{:=^50}".format(fold_title), logger=self.logger)
        logger.info(grps.data_map, logger=self.logger)
      
    # def valid_loop(self, fold):
    #     for vld_fold in fold.vld:
    #         vld_fold_path = join('valid','-'.join(vld_fold[0][1:]))
    #         if not os.path.exists(vld_fold_path):
    #             os.makedirs(vld_fold_path)
    #         og_path = self._chdir(vld_fold_path)
            
    #         # Copy model args and data config
    #         cfgs = self.cm.copy_configs()
            

    #         fold_grps = self.grps.deepcopy()
    #         leave_out = [*vld_fold, *fold.tst]
    #         trn_grp = fold_grps.data_map.drop(leave_out)
    #         trn_grp.drop('test', errors='ignore', inplace=True)
    #         vld_grp = fold_grps.data_map.loc[vld_fold].rename(index=dict(train='valid'))
    #         fold_grps.data_map = pd.concat([trn_grp, vld_grp]).sort_index()
    #         set_trace()

    #         self._train_and_valid(fold_grps, cfgs['model'], cfgs['mutate'])
            
    #         self._reset(og_path)
    
    # def tuning_loop():
    #     pass
    
    def _train_and_test(self, grps, model_args, mutate_cfg):
        self._prep(grps, model_args, mutate_cfg, save_test=True)
    
        logger.info("Beginning Training...", logger=self.logger)
        train_timer = utils.Timer()
        train_timer.start()

        trn_prebuilt = self._train(grps, get_train_args(model_args))

        train_elapsed_time = train_timer.stop()
        logger.info(f"SUCCESS: Training complete. Time to complete {train_elapsed_time}", 
                    logger=self.logger)

        logger.info("Beginning Testing...", logger=self.logger)
        test_timer = utils.Timer()
        test_timer.start()

        self._test(grps, get_test_args(model_args), prebuilt=trn_prebuilt)

        test_elapsed_time = test_timer.stop()
        logger.info(f"SUCCESS: Testing complete. Time to complete {test_elapsed_time}", 
                    logger=self.logger)
        logger.info(f"SUCCESS: LOGOCV fold complete. Time to complete {train_elapsed_time + test_elapsed_time}", 
                    logger=self.logger)

    def _prep(self, grps, model_args, mutate_cfg, save_test=False):
        model_args['seed'] = self.rng.randint(0, 9999)
        logger.info(f"Setting fold seed to {model_args['seed']}", logger=self.logger)
        set_all_seeds(model_args['seed'])
 
        train.save_configs(model_args=get_train_args(model_args), prefix="trn")
        if save_test:
            train.save_configs(model_args=get_test_args(model_args), prefix="tst")
        
        logger.info(f"Instantiating 'mutate' config...", logger=self.logger)
        built_mutate_cfg = train.instantiate_data_config(dict(mutate=mutate_cfg))
        logger.info(f"SUCCESS: 'mutate' config instantiated", logger=self.logger)
            
        logger.info("Applying mutators...", logger=self.logger)
        dutils.data.run_group_mutators(grps, built_mutate_cfg['mutate'])
        logger.info(f"SUCCESS: mutators applied.", logger=self.logger)

    def _train(self, grps, model_args):
        trn_exp_path = join(os.getcwd(), ds.TRN_DIR)
        if not os.path.exists(trn_exp_path):
            os.makedirs(trn_exp_path)
        og_path = self._chdir(trn_exp_path)
        
        trn_df, _ = train.get_train_valid_group(grps)
        tag_headers = list(trn_df.index.names)
        trn_tags = trn_df.index.unique()[0]
        
        trn_model_args, _ = train.train(
            trn=trn_df.ravel()[0],
            trn_tags=trn_tags,
            tag_headers=tag_headers,
            trn_exp_path=trn_exp_path, 
            model_args=get_train_args(model_args)
        )
        
        self._reset(og_path)
        
        return trn_model_args
           
    # TODO: Add warning if test data as been compressed
    def _test(self, grps, model_args, prebuilt=None):
        tst_exp_path = join(os.getcwd(), ds.TST_DIR)
        if not os.path.exists(tst_exp_path):
            os.makedirs(tst_exp_path)
        og_path = self._chdir(tst_exp_path)
        
        tst_df = test.get_test_group(grps)
        
        tag_headers = list(tst_df.index.names)
        tst_tags = tst_df.index.unique()

        test.test(
            tst=tst_df.ravel(),
            tst_tags=tst_tags,
            tag_headers=tag_headers,          
            tst_exp_path=tst_exp_path,
            model_args=model_args,
            prebuilt=prebuilt
        )
        
        self._reset(og_path)
    
    
    # def _train_and_valid(self, grps, model_args, mutate_cfg):
    #     trn_model_args = get_train_args(model_args)
        
    #     trn_model_args['seed'] = self.rng.randint(0, 9999)
    #     logger.info(f"Setting fold seed to {logo_args.seed}", logger=self.logger)
    #     set_all_seeds(logo_args.seed)

    #     trn_exp_path = ds.TRN_DIR
    #     if not os.path.exists(trn_exp_path):
    #         os.makedirs(trn_exp_path)
    #     og_path = self._chdir(trn_exp_path)
        
    #     logger.info(f"Instantiating 'mutate' config...", logger=self.logger)

            
    #     logger.info("Applying mutators...", logger=self.logger)
    #     dutils.data.run_group_mutators(grps, mutate_cfg)
    #     train.save_configs(model_args=logo_args.__dict__)
    #     logger.info(f"SUCCESS: mutators applied.", logger=logo_logger)

    #     # Split data
    #     trn_df, vld_df = train.get_train_valid_group(logo_grps)
    #     tst_df = test.get_test_group(logo_grps)

    #     # Train
    #     logger.info("Beginning Training...", logger=logo_logger)
    #     train_timer = utils.Timer()
    #     train_timer.start()
        
    #     tag_headers = list(trn_df.index.names)
    #     trn_tags = trn_df.index.unique()[0]
    #     vld_tags = vld_df.index.unique()[0] if vld_df is not None else None
        
    #     trn_model_args, _ = train.train(
    #         trn=trn_df.ravel()[0],
    #         vld=vld_df.ravel()[0],
    #         trn_tags=trn_tags,
    #         tag_headers=tag_headers,
    #         vld_tags=vld_tags,
    #         trn_exp_path=trn_exp_path, 
    #         model_args=logo_args.get_train_args()
    #     )
        
    #     train_elapsed_time = train_timer.stop()
    #     logger.info(f"SUCCESS: Training complete. Time to complete {train_elapsed_time}", 
    #                 logger=logo_logger)
    #     self._chdir(og_path)
        
    #     return trn_mode_args
    

            
    def generate_repeated_folds(self, mdf, logo_level):      
        groupby = mdf.loc[['train']].groupby(logo_level)
        
        if self.cm.has_tunables and len(groupby) < 3:
            err = f"Not enough training data to perform nested-CV, length is only {len(groupby)} needs 3."
            raise ValueError(err)
        elif len(groupby) < 2:
            err = f"Not enough training data to perform CV, length is only {len(groupby)} needs 2."
            raise ValueError(err)
        
        logo_folds = []
        for _, tst_df in groupby:
            folds = Folds()
            tst_idx = [*tst_df.index]

            base_tst_idx = mdf.get(['test'], pd.DataFrame()).index
            tst_idx_combine = [*tst_df.index, *base_tst_idx]
            folds.tst = tst_idx
            folds.tst_combine =  tst_idx_combine
            if self.cm.has_tunables:
                vld_folds = []
                for _, vld_df in groupby:
                    vld_idx = [*vld_df.index]
                    
                    if vld_idx == tst_idx:
                        continue
                    
                    base_vld_idx = mdf.get(['valid'], pd.DataFrame()).index
                    vld_idx_combine = [*vld_df.index, *base_vld_idx]
                    folds.vld.append(vld_idx)
                # trn_idx = [*mdf.drop([*vld_idx_combine, *tst_idx_combine]).index] 
                # folds.trn.append(trn_idx)
                # folds.vld_combine.append(vld_idx_combine)

            logo_folds.append(folds)
        return logo_folds
    
# def leave_one_group_out(data_cfg, model_args, exp_dir):
#     """ Performs leave-one-group-out (LOGO) cross-validation (CV) via dataset, task, or trials
    
#         Since this function utilizes the deepbci.data_utils.data.Groups object then
#         it will be focused on using LOGO based on the 3 out of the 4 multi-index levels
#         within Groups.data_map. These are dataset (i.e. task), subject, and trial.
        
#         WARNING: 
#             When mutating do not add a Groups.compress() method to compress based
#             on the 'group' level. This will be done automatically within the CV loop.
#             Likewise pay attention to which level you compress when using mutate. If you
#             compress a level which determines the fold (i.e. which level LOGO will
#             be based on) then an error will be thrown.
    
#     """
#     # Create save directory i.e. experiment path
#     if exp_dir is None:
#         exp_dir = ds.LOGO_EXP_DIR
#     exp_path = train.build_exp_path(exp_dir)
#     original_dir = os.getcwd()
#     os.chdir(exp_path)
    
#     # Create logger
#     logger.configure(join(exp_path, 'logs'),  ['log'], log_prefix='logo-')
#     logo_logger = logger.Logger.CURRENT
    
#     logger.info("Beginning LOGOCV...")
#     timer = utils.Timer()
#     timer.start()
    
#     train.set_default_resolvers()
#     built_data_cfg = train.instantiate_data_config(data_cfg)
#     grps = train.build_groups(groups=built_data_cfg['groups'], seed=model_args.seed)
    
#     # Save configs
#     train.save_configs(data_cfg=data_cfg, model_args=model_args.__dict__, prefix="logocv")
    
#     # Generate logo folds
#     logo_folds = LOGOFolds.generate_repeated_folds(grps.data_map, model_args.logo_level)
#     logo_logger.info(f"Folds {len(logo_folds)}")
#     # Select random seed for each fold. If seed is passed then the same seeds will be generated.
#     if model_args.seed is not None:
#         rng = np.random.RandomState(model_args.seed)
#     logo_seeds = np.random.choice(9999, len(logo_folds), replace=False)

#     logo_paths = []
#     for f, fold in enumerate(logo_folds):
#         logo_args = deepcopy(model_args)
#         # Set fold seed
#         logo_args.seed = int(logo_seeds[f])
#         logger.info(f"Setting all seeds to {logo_args.seed}", logger=logo_logger)
#         set_all_seeds(logo_args.seed)
        
#         # TODO: Add option to save base data to disk instead of creating a deep copy
#         logo_grps = grps.deepcopy()

#         # Set train/valid/test sets and combine into new df
#         logo_trn = logo_grps.data_map.loc[fold.trn]
#         logo_vld = logo_grps.data_map.loc[fold.vld].rename(index=dict(train='valid'))
#         logo_tst = logo_grps.data_map.loc[fold.tst].rename(index=dict(train='test'))
#         logo_grps.data_map = pd.concat([logo_trn, logo_vld, logo_tst]).sort_index()
        
#         # Log current fold and data
#         logo_fold_title = "LOGO - {}: {}".format(model_args.logo_level, fold.leave_out)
#         logo_fold_name = "LOGO-{}-{}".format(model_args.logo_level, fold.leave_out)
#         logger.info("{:=^50}".format(logo_fold_title))
#         logger.info(logo_grps.data_map)

#         # Build current fold directory and path
#         logo_dir = '-'.join([str(exp) for exps in fold.leave_out for exp in exps])
#         logo_path = join(exp_path, logo_dir)
#         if not os.path.exists(logo_path):
#             os.makedirs(logo_path)
#         logo_paths.append(logo_path)
#         # Change directory to current fold path
#         os.chdir(logo_path)
        
#         trn_exp_path = join(logo_path, ds.TRN_DIR)
#         if not os.path.exists(trn_exp_path):
#             os.makedirs(trn_exp_path)
            
#         tst_exp_path = join(logo_path, ds.TST_DIR)
#         if not os.path.exists(tst_exp_path):
#             os.makedirs(tst_exp_path)

#         # Run mutations on logo selected data and save model args for current fold
#         logger.info("Applying mutators...", logger=logo_logger)
#         dutils.data.run_group_mutators(logo_grps, built_data_cfg['mutate'])
#         train.save_configs(model_args=logo_args.__dict__)
#         logger.info(f"SUCCESS: mutators applied.", logger=logo_logger)

#         # Split data
#         trn_df, vld_df = train.get_train_valid_group(logo_grps)
#         tst_df = test.get_test_group(logo_grps)

#         # Train
#         logger.info("Beginning Training...", logger=logo_logger)
#         train_timer = utils.Timer()
#         train_timer.start()
        
#         tag_headers = list(trn_df.index.names)
#         trn_tags = trn_df.index.unique()[0]
#         vld_tags = vld_df.index.unique()[0] if vld_df is not None else None
        
#         trn_model_args, _ = train.train(
#             trn=trn_df.ravel()[0],
#             vld=vld_df.ravel()[0],
#             trn_tags=trn_tags,
#             tag_headers=tag_headers,
#             vld_tags=vld_tags,
#             trn_exp_path=trn_exp_path, 
#             model_args=logo_args.get_train_args()
#         )
#         train_elapsed_time = train_timer.stop()
#         logger.info(f"SUCCESS: Training complete. Time to complete {train_elapsed_time}", 
#                     logger=logo_logger)

#         # Test
#         logger.info("Beginning Testing...", logger=logo_logger)
#         test_timer = utils.Timer()
#         test_timer.start()
        
#         tag_headers = list(tst_df.index.names)
#         tst_tags = tst_df.index.unique()
        
#         test.test(
#             tst=tst_df.ravel(),
#             tst_tags=tst_tags,
#             tag_headers=tag_headers,
#             trn_exp_path=trn_exp_path,
#             tst_exp_path=tst_exp_path,
#             model_args=logo_args.get_test_args(),
#             prebuilt=trn_model_args
#         )
        
#         test_elapsed_time = test_timer.stop()
#         logger.info(f"SUCCESS: Testing complete. Time to complete {test_elapsed_time}", logger=logo_logger)
#         logger.info(f"SUCCESS: LOGOCV fold complete. Time to complete {train_elapsed_time + test_elapsed_time}", 
#                     logger=logo_logger)
        
#         # Revert logger and working directory back 
#         logger.Logger.CURRENT = logo_logger
#         os.chdir(exp_path)
        
#     generate_experiments_summary(logo_paths)
#     os.chdir(original_dir)
#     elapsed_time = timer.stop()
#     logger.info(f"SUCCESS: LOGOCV complete. Time to complete {elapsed_time}")

# class Folds():
#     def __init__(self, trn, vld, tst, leave_out):
#         self.trn = trn
#         self.vld = vld
#         self.tst = tst
#         self.leave_out = leave_out
    
# class LOGOFolds():

#     @staticmethod
#     def generate_repeated_folds(mdf, logo_level):
        
#         groupby = mdf.loc[['train']].groupby(logo_level)
        
#         if len(groupby) < 3:
#             err = f"Not enough training data to perform logocv, length is only {len(groupby)}."
#             raise ValueError(err)
        
#         logo_folds = []
#         for tst_idx, tst_df in groupby:
#             vld_folds = []
#             base_tst_idx = mdf.get(['test'], pd.DataFrame()).index
#             tst_idx_all = [*tst_df.index, *base_tst_idx]
            
#             for vld_idx, vld_df in groupby:
#                 if vld_idx == tst_idx:
#                     continue
                
#                 base_vld_idx = mdf.get(['valid'], pd.DataFrame()).index
#                 vld_idx_all = [*vld_df.index, *base_vld_idx]
                
#                 trn_idx = [*mdf.drop([*vld_idx_all, *tst_idx_all]).index] 
                
#                 folds = Folds(
#                     trn=trn_idx, 
#                     vld=vld_idx_all, 
#                     tst=tst_idx_all,
#                     leave_out=[vld_idx, tst_idx]
#                 )

#                 logo_folds.append(folds)
#         return logo_folds
    
def get_train_args(args):
    get_keys = TrainArgs.__dataclass_fields__.keys()
    args_found = _get_args(args, get_keys)

    return args_found

def get_test_args(args):
    get_keys = TestArgs.__dataclass_fields__.keys()
    args_found = _get_args(args, get_keys)

    return args_found

def _get_args(target_dict, get_keys):
    new_args = {}
    for key in get_keys:
        value = target_dict.get(key)
        if value is not None:
            new_args[key] = value
            
    return new_args

@dataclass(order=True)
class _LOGOCVDefaultArgs():
    """ Default args for leave-one-group-out classification training.

        Args:
            logo_level (list):
    """
    logo_level: list = field(default_factory=lambda: ['dataset', 'subject', 'trial'])
    
    def __getitem__(self, item):
        return getattr(self, item)

@dataclass(order=True)
class LOGOCVArgs(_LOGOCVDefaultArgs,
                 _TrainDefaultArgs,
                 _TestDefaultArgs, 
                 _TrainRequiredArgs,
                 _TestRequiredArgs):
    """ Class which combines both LOGO, train and test args. """
    
    def __getitem__(self, item):
        return getattr(self, item)

