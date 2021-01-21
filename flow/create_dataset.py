from sklearn import datasets


def create_artificial_dataset(file_name: str, n_samples: int):
    x, y = datasets.make_regression(n_samples=n_samples,
                                    n_features=1,
                                    n_informative=1,
                                    n_targets=1,
                                    bias=3.0,
                                    noise=1.0)
    # just make sure data is in the right format, i.e. one feature
    assert x.shape[1] == 1
    # write to file, tab separated
    with open(file_name, "w") as _file:
        for _ in range(len(x)):
            _file.write("{}\t{}\n".format(x[_][0], y[_]))

    return


if __name__ == '__main__':
    create_artificial_dataset(file_name="dataset.txt", n_samples=1000)