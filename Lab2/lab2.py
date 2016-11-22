from ml.pca import pca


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

        pca(data)


if __name__ == '__main__':
    main()
