import numpy as np

def load(X_file,y_file=None,ids_file=None,seed=12345):
    """
    Load training data from .npy files.
    """
    X = np.load(X_file)
    X = X.astype(np.float32)
    X /= 255

    if y_file:
        y = np.load(y_file)

        if not seed:
            seed = np.random.randint(1, 10e6)
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        return X,y

    else:
        ids = np.load(ids_file)

        return X, ids
