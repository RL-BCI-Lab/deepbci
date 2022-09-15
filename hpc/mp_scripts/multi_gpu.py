import tensorflow as tf
import time
import os

# def multi_gpu(func):
#     def _wrapper(*args, **kwargs):
#         print(func, kwargs, args)
#         gpu = kwargs.pop('gpu', None)
       
#         print(gpu)
#         physical_gpus = tf.config.list_physical_devices('GPU')
#         tf.config.experimental.set_visible_devices(physical_gpus[gpu], 'GPU')
#         tf.config.experimental.set_memory_growth(physical_gpus[gpu], True)
        
#         return func(*args, **kwargs)
#     return _wrapper


def train(x, steps):
    # gpus = tf.config.list_logical_devices('GPU')
    # print(gpus)
    x = tf.random.normal(x)
    x = x @ tf.transpose(x)
    start = time.time()
    for i in range(steps):
        x = x @ tf.transpose(x)
    _ = x.numpy()
    end = time.time()
    print(os.getpid(), ' returning: ', end - start)
    return end - start

def multi_gpu(func, gpu_queue, *args, **kwargs):
    # Wrap in try-catch so errors will actually get printed
    try:
        # Get gpu for this process
        gpu = gpu_queue.get()
        
        # Set GPU visible device and set memory growth
        physical_gpus = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(physical_gpus[gpu], 'GPU')
        tf.config.experimental.set_memory_growth(physical_gpus[gpu], True)
        
        # Run function given selected gpu
        logical_gpus = tf.config.list_logical_devices('GPU')
        with tf.device(logical_gpus[0]):
            results = func(*args, **kwargs)
            
        # Add gpu back to being available 
        gpu_queue.put(gpu)
    except Exception as e:
        print(e)
        return 
    
    return results

def multi_gpu2(func, gpu, *args, **kwargs):

    physical_gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(physical_gpus[gpu], 'GPU')
    tf.config.experimental.set_memory_growth(physical_gpus[gpu], True)
    
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(logical_gpus[0])
    with tf.device(logical_gpus[0]):
        results = func(*args, **kwargs)
    
    return results

if __name__ == "__main__":
    import multiprocessing as mp
    import numpy as np

    shape = (10000, 500)
    steps = 10
    NUM_PROC = mp.cpu_count()
    NUM_GPUS = 2
    manager = mp.Manager()
    gpu_queue = manager.Queue(maxsize=NUM_GPUS)
    for i in range(NUM_GPUS):
        gpu_queue.put(i)

    default_kwargs = dict(func=train, gpu_queue=gpu_queue)
    kwargs = [
        dict(**default_kwargs, x=shape, steps=10),
        dict(**default_kwargs, x=shape, steps=30),
        dict(**default_kwargs, x=shape, steps=60),
        dict(**default_kwargs, x=shape, steps=80)
    ]
        
    pool = mp.Pool()
    results = []
    log_results = lambda result: results.append(result)
    jobs = []
    for kws in kwargs:
        jobs.append(pool.apply_async(multi_gpu, kwds=kws, callback=log_results))
    pool.close()

    start = time.time()
    while True:
        check_jobs = [j.ready() for j in jobs]
        if not all(check_jobs):
            jobs_left = len(jobs) - sum(check_jobs)
            print('Time: {:4f} Tasks Left: {}'.format(time.time() - start, jobs_left))
            time.sleep(1)
        else:
            break

    pool.join()
    print([j.successful() for j in jobs])
    print("Results: {}".format(results))