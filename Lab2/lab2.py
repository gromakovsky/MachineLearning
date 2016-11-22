import numpy as np

from ml.pca import pca_eigens, broken_stick, transformation_matrix


FILES = [
    'data/newBasis1',
    'data/newBasis2',
    'data/newBasis3',
]


def main():
    for file in FILES:
        print('Current file:', file)
        with open(file) as f:
            data = [[float(i) for i in s.split()] for s in f.readlines()]

        arr = np.array(data)
        eigens = pca_eigens(arr)
        k = broken_stick(eigens)
        print('Number of principal components:', k)
        m = transformation_matrix([e[1] for e in eigens], k)
        transformed = (m @ arr.T).T
        with open(file + '-transformed', mode='w') as out:
            for item in transformed:
                out.write(" ".join([str(i.real) for i in item]))
                out.write('\n')


if __name__ == '__main__':
    main()
