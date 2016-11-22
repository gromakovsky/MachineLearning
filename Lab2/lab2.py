import numpy as np

from ml.pca import pca_eigens, broken_stick, transformation_matrix, BrokenStickRes

FILES = [
    'data/newBasis1',
    'data/newBasis2',
    'data/newBasis3',
]


def visualize_broken_stick(broken_stick_res: BrokenStickRes):
    pass


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
