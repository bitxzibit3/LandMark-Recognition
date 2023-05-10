import time
import sys


def memory_counter(ex, all=False):
    """
    Memory counter for existing class instance
    all - count all variables and methods in ex, else exclude __methods__
    """
    mem = 0
    if all:
        for key, val in ex.__dict__.items():
            mem += sys.getsizeof(val)
        return mem
    else:
        for key, val in ex.__dict__.items():
            if key.startswith('_'):
                continue
            else:
                mem += sys.getsizeof(val)
        return mem


def dataset_metric(cls, print_info=True, **kwargs):
    """
    Comparing of classes with datasets: init, traverse, memory
    """
    print(f'Class name: {cls.__name__}')
    begin = time.time()
    ex = cls(**kwargs)
    to_init = time.time() - begin
    print('Time to init: {:.5f} s'.format(to_init))
    begin = time.time()
    for _ in ex:
        pass
    to_traverse = time.time() - begin
    print('Time to traverse: {:.5f} s'.format(to_traverse))
    memory = memory_counter(ex)
    info = '\n'.join(['Memory: {} bytes = {:.3f} MB',
                      'Total elements: {} elements',
                      'Mean iteration time: {:.4f} s',
                      'Mean memory usage per element: {:.4f} bytes',
                      '']).format(memory, memory / 10 ** 6,
                                  len(ex),
                                  to_traverse / len(ex),
                                  memory / len(ex))
    if print_info:
        print(info)

    d = (cls.__name__, to_init, to_traverse, memory)
    del ex
    return d
