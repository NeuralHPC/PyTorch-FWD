import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import tqdm


import time
from src.util import get_batched_celebA_paths, batch_loader, get_label_dict

def test_celebA_loader():
    labels_dict = get_label_dict('/home/wolter/uni/diffusion/data/celebA/CelebA/Anno/identity_CelebA.txt')
    path_batches = get_batched_celebA_paths(64)
    start = time.perf_counter()
    image_batch, image_labels = batch_loader(path_batches[0], labels_dict)
    end = time.perf_counter()
    print(end - start)
    assert image_batch.shape == (65, 64, 64, 3)
    assert image_labels.shape == (65,)


def test_multibatch_loader():
    labels_dict = get_label_dict('/home/wolter/uni/diffusion/data/celebA/CelebA/Anno/identity_CelebA.txt')
    path_batches = get_batched_celebA_paths(1000)
    epoch_start = time.perf_counter()
    total = len(path_batches)
    # We can use a with statement to ensure threads are cleaned up promptly
    batch_loader_w_dict = partial(batch_loader, labels_dict=labels_dict)
    with ThreadPoolExecutor() as executor:
        load_asinc_dict = {executor.submit(batch_loader_w_dict, path_batch): path_batch
                         for path_batch in path_batches}
        start = time.perf_counter()
        for done, future in enumerate(as_completed(load_asinc_dict)):
            path = load_asinc_dict[future]
            data = future.result()
            end = time.perf_counter()
            print(done/total*100, path[0].split('/')[-1], data[0].shape, data[1].shape, end - start)
    epoch_end = time.perf_counter()
    print("total time spent loading in epoch", epoch_end - epoch_start, "[s]", (epoch_end - epoch_start)/60., "[min]" )
    print((epoch_end - epoch_start)/len(path_batches), 'seconds per batch')
    assert data[0] == (1000, 64, 64, 3)
