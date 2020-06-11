import os
import subprocess

if __name__ == "__main__":
    path = "<PATH TO RESULTS>"

    dirs = os.listdir(path)
    for dir in dirs:
        if dir != ".DS_Store":
            p1 = subprocess.Popen("python hyperparameter_plotting.py {}".format(os.path.join(path, dir)).split(" "))
            p1.wait()