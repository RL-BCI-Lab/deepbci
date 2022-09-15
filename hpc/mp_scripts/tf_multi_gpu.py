import time
import os
import multiprocessing as mp
from pdb import set_trace

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Notes:
# - Wrapping the multi_gpu function doesnt work with multi procresses
# - For tensorflow, all model creation must be done AFTER tf_multi_gpu has been 
#   called otherwise tensroflow will throw an error sine memory growth can not be set AFTER
#   initialization.
# - You can not pass tensroflow objects in the config otherwise multi-process will
#   silently fail with no reasons!

def run_fiteval_multi_gpu(cfgs, n_gpus):
    """ Runs multiple basic training and then testing experiments """
    jobs = []
    results = []
    log_results = lambda result: results.append(result)

    # Build available queue which contains GPUs which are aviable for use
    manager = mp.Manager()
    avail_gpus = manager.Queue(maxsize=n_gpus)
    for i in range(n_gpus):
        avail_gpus.put(i)

    pool = mp.Pool(n_gpus)

    for cfg in cfgs:
        (cfg_name, kwargs), = cfg.items()

        # Add additional kwargs
        kwargs['avail_gpus'] = avail_gpus
        kwargs['func'] = fake_script
        # tf_multi_gpu(**kwargs)

        # Initialize processes
        job = pool.apply_async(tf_multi_gpu, 
                               kwds=kwargs, 
                               callback=log_results)
        jobs.append(job)
        
    job_status(jobs)
    
    pool.close()
    pool.join()
    print("Successful Jobs: {} ".format([j.successful() for j in jobs]))
    print(results)

def tf_multi_gpu(func, avail_gpus, *args, **kwargs):
    # Wrap in try-catch so errors will actually get printed
    try:
        # Get gpu for this process
        gpu = avail_gpus.get()
   
        # Set GPU visible device and set memory growth
        physical_gpus = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(physical_gpus[gpu], 'GPU')
        tf.config.experimental.set_memory_growth(physical_gpus[gpu], True)
 
        # Run function given selected gpu
        logical_gpus = tf.config.list_logical_devices('GPU')
        with tf.device(logical_gpus[0]):
            results = func(*args, **kwargs)
            
        # Add gpu back to being available 
        avail_gpus.put(gpu)
        
    except Exception as e:
        raise Exception(e)
    
    return results

def job_status(jobs):
    import psutil
    import time
    start = time.time()
    while True:
        check_jobs = [j.ready() for j in jobs]
        if not all(check_jobs):
            jobs_left = len(jobs) - sum(check_jobs)
            print('Time: {:4f} Tasks Left: {} Memory: {}'.format(time.time() - start, 
                                                                jobs_left,
                                                                psutil.virtual_memory()[2]))
            time.sleep(1)
        else:
            break

def fake_script(use_model, metrics=['accuracy'], epochs=1):
    
    mnist = tf.keras.datasets.mnist
    (X_trn, y_trn), (X_tst, y_tst) = mnist.load_data()
    X_trn, X_tst = X_trn / 255.0, X_tst / 255.0

    if use_model.lower() == 'fc':
        model = tf.keras.Sequential(
            [   layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(10, activation="softmax"),
            ]
        )
    elif use_model.lower() == 'cnn':
        model = tf.keras.Sequential(
            [
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(), 
                layers.Dense(64, activation='relu'),
                layers.Dense(10, activation="softmax")
            ]
        )

    loss = tf.keras.losses.sparse_categorical_crossentropy,
    optim = tf.keras.optimizers.Adam(.01)
    model.compile(optimizer=optim, loss=loss, metrics=metrics) 
    
    hist = model.fit(X_trn[..., None], y_trn, epochs=epochs, batch_size=1024, verbose=0)

    # Clear tf/keras graph state
    tf.keras.backend.clear_session()
    
    tst_loss, tst_acc = model.evaluate(X_tst[..., None], y_tst, verbose=0)

    # Clear tf/keras graph state
    tf.keras.backend.clear_session()
    
    return tst_loss, tst_acc   

if __name__ == "__main__":
    
    cfgs=[
        {'exp1': {
            'use_model': 'fc',
            'epochs': 1
        
        }},
        {'exp2': {
            'use_model': 'cnn',
            'epochs': 1
        }}
    ]
    
    run_fiteval_multi_gpu(cfgs, n_gpus=1)