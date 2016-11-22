import numpy as np
import matplotlib.pyplot as plt

from ml.pca import pca_eigens, broken_stick, transformation_matrix, BrokenStickRes

FILES = [
    'data/newBasis1',
    'data/newBasis2',
    'data/newBasis3',
]


def visualize_broken_stick(broken_stick_res: BrokenStickRes):
    k = broken_stick_res.k
    to_show = k + 2
    eigen_values = broken_stick_res.normalized[:to_show]
    eigen_bar = plt.bar(np.arange(to_show), eigen_values, width=0.5)
    steps = broken_stick_res.steps[:to_show]
    steps_bar = plt.bar(np.arange(to_show), steps, width=0.3, color='red')
    plt.legend((eigen_bar, steps_bar), ('Normalized eigen values', 'Broken stick steps'))
    plt.show()


def main():
    for file in FILES:
        print('Current file:', file)
        with open(file) as f:
            data = [[float(i) for i in s.split()] for s in f.readlines()]

        arr = np.array(data)
        eigens = pca_eigens(arr)
        broken_stick_res = broken_stick(eigens)
        assert isinstance(broken_stick_res, BrokenStickRes)
        print('Number of principal components:', broken_stick_res.k)
        visualize_broken_stick(broken_stick_res)
        m = transformation_matrix([e[1] for e in eigens], broken_stick_res.k)
        transformed = (m @ arr.T).T
        with open(file + '-transformed', mode='w') as out:
            for item in transformed:
                out.write(" ".join([str(i.real) for i in item]))
                out.write('\n')


if __name__ == '__main__':
    main()
