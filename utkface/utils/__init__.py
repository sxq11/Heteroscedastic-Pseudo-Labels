import os
import sys
import errno
import typing
import datetime
import cv2  
import matplotlib
import numpy as np
import torch
import tqdm


def get_mean_and_std(dataset: torch.utils.data.Dataset,
                     samples: int = 128,
                     batch_size: int = 8,
                     num_workers: int = 4):

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    n = 0  
    s1 = 0.  
    s2 = 0.  

    for (x,_,*_) in tqdm.tqdm(dataloader):  
        x = x.transpose(0, 1).contiguous().view(3, -1)  
        n += x.shape[1]
        s1 += torch.sum(x, dim=1).numpy()
        s2 += torch.sum(x ** 2, dim=1).numpy()
    mean = s1 / n  
    std = np.sqrt(s2 / n - mean ** 2) 

    mean = mean.astype(np.float32)  
    std = std.astype(np.float32)  

    return mean, std


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            try:
                os.makedirs(os.path.dirname(fpath))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


__all__ = ["get_mean_and_std", "Logger"]
