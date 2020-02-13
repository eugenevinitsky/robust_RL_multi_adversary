"""Construct plots of how we expect the eigenvalues to evolve with dimension. What fraction are stable vs.
unstable as we increase the dimension?"""
import os

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def plot_eigenvals(output_path):
    perturb_scale = 0.4
    dims = [2, 4, 8]
    num_samples = 500
    eigs = [[] for i in range(len(dims))]
    for dim_idx, dim in enumerate(dims):
        num_stable = 0
        for i in range(num_samples):
            new_mat = (perturb_scale / dim) * np.random.uniform(low=-1, high=1, size=(dim, dim))
            eigvals = np.linalg.eigvals(new_mat)
            for eigval in eigvals:
                if np.linalg.norm(eigval) < (perturb_scale / 2):
                    num_stable += 1
                eigs[dim_idx].append([np.real(eigval), np.imag(eigval)])
        print('fraction stable of dim {} is {}'.format(dim, num_stable / (num_samples * dim)))

    # create a bar plot of the magnitudes
    plt.figure()
    norms = [np.mean(np.linalg.norm(eig_list, axis=-1)) for eig_list in eigs]
    plt.bar(np.arange(len(dims)), norms)
    plt.xticks(np.arange(len(dims)), [str(dim) for dim in dims])
    plt.savefig(os.path.join(output_path, 'dim_mag.png'))

    # create a circle plot of magnitudes
    plt.figure()
    colors = cm.rainbow(np.linspace(0, 1, len(dims)))
    for dim_idx, dim in enumerate(dims):
        eigs_arr = np.array(eigs[dim_idx])
        plt.scatter(eigs_arr[:, 0], eigs_arr[:, 1], color=colors[dim_idx])

    plt.xlabel('Real axis')
    plt.ylabel('Imaginary Axis')
    plt.title('Eigenvalues of random matrices as dimension increases')

    plt.legend([str(dim) for dim in dims])
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'eig_scatter.png'))


if __name__ == '__main__':
    plot_eigenvals(os.path.dirname(os.path.abspath(__file__)))
