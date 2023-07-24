import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool

import time
from src.util import get_batched_celebA_paths, batch_loader, multi_batch_loader

def test_celebA_loader():
    path_batches = get_batched_celebA_paths(64)
    start = time.perf_counter()
    image_batch = batch_loader(path_batches[0])
    end = time.perf_counter()
    print(end - start)
    assert image_batch.shape == (65, 218, 178, 3)


def test_multibatch_loader():
    path_batches = get_batched_celebA_paths(64)
    start = time.perf_counter()
    image_batch = multi_batch_loader(path_batches[:20])
    end = time.perf_counter()
    print(end - start)
    pass