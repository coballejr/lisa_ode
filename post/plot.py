import matplotlib.pyplot as plt
import os
import numpy as np
import glob

class Post:

    def __init__(self, out_dir: str):
        self.out_dir = out_dir

    def load(self, loss: str = 'valid'):

        losses = {}
        os.chdir(self.out_dir)

        directories = filter(os.path.isdir, os.listdir())
        for exp_dir in directories:
            filename = glob.glob(exp_dir+'/'+loss+'_fc*.npy')
            case = exp_dir.split('_')[-1]
            results = np.load(filename[0])
            losses[case] = results

        return losses

    def plot(self, losses: dict, fname: str, **kwargs):
        for case, results in losses.items():
            iteration, mse = results[:,0], results[:,1]
            plt.plot(iteration, mse, label = case)
        plt.legend()
        plt.savefig(fname)
        plt.close()



