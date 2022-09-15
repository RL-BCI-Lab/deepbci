import os
import ast
import re
import time
import functools
import importlib
from os.path import join
from inspect import getmembers, ismethod
from pdb import set_trace

import yaml
import dill
import numpy as np

import deepbci as dbci

def get_module_root():
    return os.path.dirname(dbci.__file__)

def strings_match(string1, string2):
    return string1 == string2

def group_consecutive(data, gap=1):
    """
    Groups data based on consecutive numbers with specified gap

    Attributes
    ----------
    data        Data you wish to group

    gap         Allowed gap between consecutive samples to count as being consecutive
    """
    return np.split(data, np.where(np.diff(data) != gap)[0]+1)

def bordered(text):
    lines = text.splitlines()
    width = max(len(s) for s in lines)
    res = ['┌' + '─' * width + '┐']
    for s in lines:
        res.append('│' + (s + ' ' * width)[:width] + '│')
    res.append('└' + '─' * width + '┘')

    return '\n'.join(res)

def timeme(func):
    """
        Decoractor for tracking the runtime of a decorated function.

        Args:
            func (function): A Python function.
    """

    @functools.wraps(func)
    def wrapper_time(*args, **kwargs):
        start = time.monotonic()
        output = func(*args, **kwargs)
        total_time = (time.monotonic() - start)
        speed = "Run time of {}: {:.9f} seconds".format(func.__name__, total_time)
        print(bordered(text=speed))

        return output

    return wrapper_time

def printme(func):
    @functools.wraps(func)
    def wrapper_print(*args, **kwargs):
        print("{:-^50s}".format(func.__name__))
        output = func(*args, **kwargs)

        return output
        
    return wrapper_print

def get_timestamp():
    """Get timestamp in hh:mm:ss:msms and in milliseconds"""
    import datetime
    ts = datetime.datetime.now().time()
    ms = ts.hour*3600 + ts.minute*60 + ts.second + ts.microsecond/1000000
    
    return ts, ms

def parse_trials(trials):
    trial_numbers = []
    for t in trials:
        if isinstance(t, list):
            trial_numbers += np.arange(t[0], t[1]+1).tolist()
        else:
            trial_numbers.append(t)

    return np.array(trial_numbers)

def tf_allow_growth():
    import tensorflow as tf
    # Turn on allow growth so tensorflow doesnt eat all gpu vram
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available!"
    for d in physical_devices:
        tf.config.experimental.set_memory_growth(d, True)
    
def run_methods(bound_class, methods, logger=print):
    class_methods = dict(getmembers(bound_class, predicate=ismethod))
    
    for method in methods:
        method_name, kwargs = next(iter(method.items()))
        kwargs = {} if kwargs is None else kwargs
        if method_name in class_methods:
            logger("Running {}...".format(method_name))
            
            # Extract method and call
            method_call = class_methods[method_name]
            method_call(**kwargs)
            
# def make_new_method(self, method_name):
#     def new_method(self, *args, **kwargs):
#         method = getattr(self._type, method_name)
#         return method(self._obj, *args, **kwargs)
#     return new_method

# class Pointer():   
#     def __init__(self, value, name=''):
#         self._value = value
#         self._name = name
#         self._type = type(obj)
        
#         for attr_name in dir(self._type):
#             if attr_name not in dir(Pointer):
#                 if callable(getattr(self._type, attr_name)):
#                     new_attr = make_new_method(self, attr_name)
#                     setattr(Pointer, attr_name, new_attr)
#                 else:
#                     attr_value = getattr(self._value, attr_name)
#                     setattr(self, attr_name, attr_value)
    
#     def dump_yaml(self):
#         return dict(obj=self._value, name=self._name)
    
#     @property
#     def __class__(self):
#         return self._type
    
#     def set(self, value):
#         if self._type is type(value):
#             self._value = value
#         else:
#             self.__init__(value)
            
#     def __repr__(self):
#         return f"Pointer({repr(self._obj)})"
    
def get_yaml_loader():
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u"""^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""", re.X),
        list(u'-+0123456789.'))
    
    # def select_constructor(loader, node, deep=False):
    #     classname = node.__class__.__name__
    #     if classname == "SequenceNode":
    #         return loader.construct_sequence(node, deep=deep)
    #     elif classname == "MappingNode":
    #         return loader.construct_mapping(node, deep=deep)
    #     else:
    #         return loader.construct_scalar(node)
    
    # def pointer_constructor(loader, node):
    #     args = select_constructor(loader, node, deep=True)
    #     if isinstance(args, list):
    #         return Pointer(*args)
    #     elif isinstance(args, dict):
    #         return Pointer(**args)
  
    def eval_constructor(loader, node):
        """ Extract the matched value, expand env variable, and replace the match """
        value = node.value

        if isinstance(value, str):
            match = "".join(eval_matcher.findall(value))
            return eval(match)
        
    def tuple_constructor(loader, node):                                                               
        def parse_tuple(element):                                          
            if element.isdigit(): 
                return int(element)
            try:
                return float(element)
            except ValueError:
                pass 
            try:
                if ast.literal_eval(value) is None:
                    return None
            except ValueError:
                return value

        value = loader.construct_scalar(node)
        # Match tuple(*) and remove it from string. At the same time strip any whitespace
        # and split string into list based on commas.                                                                                                                                                    
        match = "".join(tuple_matcher.findall(value)).replace(' ', '').split(',')
        # Remove tailing space if tuple was formated with tailing comma tuple(*,)                                                                                                                                                   
        if match[-1] == '':                                                                                                       
            match.pop(-1)
        # Convert string to int, float, or string.                                                                                      
        return tuple(map(parse_tuple, match))                                                       
    
    def none_constructor(loader, node):
        """ Extract the matched value, expand env variable, and replace the match """
        value = node.value

        if isinstance(value, str):
            try:
                if ast.literal_eval(value) is None:
                    return None
            except ValueError:
                return value
        

    eval_matcher = re.compile(r'eval\(([^}^{]+)\)')
    loader.add_implicit_resolver('!eval', eval_matcher, None)
    loader.add_constructor(u'!eval', eval_constructor)

    tuple_matcher = re.compile(r'\(([^}^{]+)\)')
    loader.add_implicit_resolver('!tuple', tuple_matcher, None)
    loader.add_constructor(u'!tuple', tuple_constructor)

    none_matcher = re.compile(r'None')
    loader.add_implicit_resolver('!none', none_matcher, None)
    loader.add_constructor(u'!none', none_constructor)
        
    # pointer_matcher = re.compile(r'Pointer\(([^}^{]+)\)')
    # loader.add_implicit_resolver('!Pointer', pointer_matcher, None)
    # loader.add_constructor(u'!Pointer', pointer_constructor)
    
    return loader

def get_yaml_dumper():
    dumper = yaml.Dumper
    
    def ndarray_rep(dumper, data):
        return dumper.represent_list(data.tolist())

    # def pointer_rep(dumper, data):
    #     kwargs = data.dump_yaml()
    #     return dumper.represent_mapping(u'!Pointer', kwargs)
    
    dumper.add_representer(np.ndarray, ndarray_rep)
    # dumper.add_representer(Pointer, pointer_rep)
    
    return dumper

def load_yaml(yaml_path, loader=None, **kwargs):
    """ Loads a yaml file
    
        Args:
            yaml_path (str): File path to yaml file.
            
            loader (yaml.Loader): yaml loader to be used for loading. Will default
                to custom SafeLoader.
            
            kwargs (dict): Arguments for get_yaml_loader() function which
                creates the custom SafeLoader. 

        Returns:
            dict: Returns a parsed dictionary extracted from the yaml file.
    """
    
    if not os.path.exists(yaml_path):
        raise ValueError("Path to config does not exist {}".format(yaml_path))
    
    loader = get_yaml_loader(**kwargs) if loader is None else loader
    with open(yaml_path, 'r') as stream:
        params = yaml.load(stream, Loader=loader)

    return params

def dump_yaml(data, yaml_path, **kwargs):
    with open(yaml_path, 'w') as stream:
        yaml.dump(data, stream, Dumper=get_yaml_dumper(), **kwargs)

def path_to(root_dir, target, topdown=True):
    """ Search directory for a LOCAL target file/folder and construct an absolute 
        path to target.
    """
    if isinstance(target, list):
        if target[0] != os.path.sep:
            target = target.insert(0, os.path.sep)
        target = os.path.join(*target)
        # target = target.split(os.path.sep)

    for root, folders, files in os.walk(root_dir, topdown=topdown):
        found = folders + files
        for name in found:
            abs_path = os.path.join(root, name)
            # print(abs_path, target)
            # Check is abs_path "matches" target string
            if abs_path.find(target) != -1:
                return os.path.normpath(abs_path)
            
    raise FileNotFoundError("Target {} was not found in {}".format(target, root_dir))

def log_config(config, logger, spacing="",):
    for k, v in config.items():
        if isinstance(v, dict):
            logger.log("{}{}:".format(spacing, k))
            log_config(v, logger, "\t")
        else:
            logger.log("{}{}: {}".format(spacing, k, v))

def import_module(module_path: str):
    module_name, attr = module_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    if hasattr(module, attr):
        return getattr(module, attr)
    else:
        err = f"No module named '{module_name}.{attr}'"
        raise ModuleNotFoundError(err)

def pickle_dump(contents, filepath):
        with open(filepath, 'wb') as stream:
            dill.dump(contents, stream)

def pickle_load(filepath):      
    with open(filepath, 'rb') as stream:
        contents = dill.load(stream)

    return contents

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:

            raise TimerError(f"Timer is running. Use .stop() to stop it")


        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""

        if self._start_time is None:
            raise TimerError(f"Timer is not running. Make sure to call .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        
        return elapsed_time
