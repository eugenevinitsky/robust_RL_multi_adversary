import os
import subprocess

if __name__ == "__main__":
    path = "/Users/eugenevinitsky/Desktop/Research/Data/sim2real/transfer_results/adv_robust/01-31-2020"

    dirs = os.listdir(path)
    for dir in dirs:
        if dir != ".DS_Store":
            p1 = subprocess.Popen("python hyperparameter_plotting.py {}".format(os.path.join(path, dir)).split(" "))
            p1.wait()