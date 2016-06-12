import numpy as np
import sys

from framework.utils import crps, real_to_cdf
from augmentations import sampling_augmentation


def split_data(X, y, split_ratio=0.2):
    """
    Split data into training and testing.
    :param X: X
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    split = X.shape[0] * split_ratio
    X_test = X[:split, :, :, :]
    y_test = y[:split]
    X_train = X[split:, :, :, :]
    y_train = y[split:]

    return X_train, y_train, X_test, y_test    

def do_calc_crps(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    batch_size,
    loss,
    val_loss):
    print('Evaluating CRPS...')
    pred = model.predict(X_train, batch_size=batch_size, verbose=1)
    val_pred = model.predict(X_test, batch_size=batch_size, verbose=1)

    # CDF for train and test data (actually a step function)
    cdf_train = real_to_cdf(y_train)
    cdf_test = real_to_cdf(y_test)

    # CDF for predicted data
    cdf_pred = real_to_cdf(pred, loss)
    cdf_val_pred = real_to_cdf(val_pred, val_loss)

    # evaluate CRPS on training data
    crps_train = crps(cdf_train, cdf_pred)
    print('CRPS(train) = {0}'.format(crps_train))

    # evaluate CRPS on test data
    crps_test = crps(cdf_test, cdf_val_pred)
    print('CRPS(test) = {0}'.format(crps_test))

def train_single(
    model,
    X,
    y,
    f_preprocess,
    f_augmentations,
    output_weights,
    output_best_weights,
    output_val_loss,
    split_ratio=0.2,
    nb_iter=400,
    batch_size=64
    ):
    """
    Training systole and diastole models.
    """


    print('Pre-processing images...')
    for f_pre in f_preprocess:
        X = f_pre(X)

    # split to training and test
    X_train, y_train, X_test, y_test = split_data(X, y, split_ratio)

    calc_crps = 5  # calculate CRPS every n-th iteration (set to 0 if CRPS estimation is not needed)

    min_val_loss = sys.float_info.max

    print('-'*50)
    print('Training...')
    print('-'*50)

    for i in range(nb_iter):
        print('-'*50)
        print('Iteration {0}/{1}'.format(i + 1, nb_iter))
        print('-'*50)

        print('Augmenting images')
        X_train_aug = X_train.copy()
        for f_aug in f_augmentations:
            X_train_aug = f_aug(X_train_aug)

        print('Fitting model...')
        hist = model.fit(X_train_aug, y_train, shuffle=True, nb_epoch=1,
                                         batch_size=batch_size, validation_data=(X_test, y_test))

        # sigmas for predicted data, actually loss function values (RMSE)
        loss = hist.history['loss'][-1]
        val_loss = hist.history['val_loss'][-1]

        if (calc_crps > 0 and i % calc_crps == 0) or (val_loss < min_val_loss):
            do_calc_crps(model,X_train,y_train,X_test,y_test,batch_size,loss,val_loss)

        print('Saving weights...')
        # save weights so they can be loaded later
        model.save_weights(output_weights, overwrite=True)

        # for best (lowest) val losses, save weights
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            model.save_weights(output_best_weights, overwrite=True)

        # save best (lowest) val losses in file (to be later used for generating submission)
        with open(output_val_loss, mode='w+') as f:
            f.write(str(min_val_loss))

def train_single_sampling(
    model,
    X,
    y,
    f_preprocess,
    f_augmentations,
    output_weights,
    output_best_weights,
    output_val_loss,
    split_ratio=0.2,
    nb_iter=400,
    batch_size=64,
    n_samples = 5,
    ):
    """
    Training systole and diastole models.
    """

    print('Pre-processing images...')
    for f_pre in f_preprocess:
        X = f_pre(X)

    # split to training and test
    X_train, y_train, X_test, y_test = split_data(X, y, split_ratio)

    calc_crps = 5  # calculate CRPS every n-th iteration (set to 0 if CRPS estimation is not needed)

    min_val_loss = sys.float_info.max

    print('-'*50)
    print('Training...')
    print('-'*50)

    for i in range(nb_iter):
        print('-'*50)
        print('Iteration {0}/{1}'.format(i + 1, nb_iter))
        print('-'*50)

        print('Augmenting images - sampling')
        X_train_sampled = sampling_augmentation(X_train,n_samples)
        X_test_sampled = sampling_augmentation(X_test,n_samples)

        print('Augmenting images')
        X_train_aug = X_train_sampled.copy()
        for f_aug in f_augmentations:
            X_train_aug = f_aug(X_train_aug)

        print('Fitting model...')
        hist = model.fit(X_train_aug, y_train, shuffle=True, nb_epoch=1,
                                         batch_size=batch_size, validation_data=(X_test_sampled, y_test))

        # sigmas for predicted data, actually loss function values (RMSE)
        loss = hist.history['loss'][-1]
        val_loss = hist.history['val_loss'][-1]

        if (calc_crps > 0 and i % calc_crps == 0) or (val_loss < min_val_loss):
            do_calc_crps(model,X_train_sampled,y_train,X_test_sampled,y_test,batch_size,loss,val_loss)

        print('Saving weights...')
        # save weights so they can be loaded later
        model.save_weights(output_weights, overwrite=True)

        # for best (lowest) val losses, save weights
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            model.save_weights(output_best_weights, overwrite=True)

        # save best (lowest) val losses in file (to be later used for generating submission)
        with open(output_val_loss, mode='w+') as f:
            f.write(str(min_val_loss))

def train_dual(
    model_systole,
    model_diastole,
    X,
    y,
    f_preprocess,
    f_augmentations,
    output_weights_systole,
    output_weights_diastole,
    output_best_weights_systole,
    output_best_weights_diastole,
    output_val_loss,
    split_ratio=0.2,
    nb_iter=200,
    batch_size=32):
    """
    Training systole and diastole models.
    """

    print('Pre-processing images...')
    for f_pre in f_preprocess:
        X = f_pre(X)

    # split to training and test
    X_train, y_train, X_test, y_test = split_data(X, y, split_ratio)

    calc_crps = 0  # calculate CRPS every n-th iteration (set to 0 if CRPS estimation is not needed)

    min_val_loss_systole = sys.float_info.max
    min_val_loss_diastole = sys.float_info.max

    print('-'*50)
    print('Training...')
    print('-'*50)

    for i in range(nb_iter):
        print('-'*50)
        print('Iteration {0}/{1}'.format(i + 1, nb_iter))
        print('-'*50)

        print('Augmenting images')
        X_train_aug = X_train.copy()
        for f_aug in f_augmentations:
            X_train_aug = f_aug(X_train_aug)

        print('Fitting systole model...')
        hist_systole = model_systole.fit(X_train_aug, y_train[:, 0], shuffle=True, nb_epoch=1,
                                         batch_size=batch_size, validation_data=(X_test, y_test[:, 0]))

        print('Fitting diastole model...')
        hist_diastole = model_diastole.fit(X_train_aug, y_train[:, 1], shuffle=True, nb_epoch=1,
                                           batch_size=batch_size, validation_data=(X_test, y_test[:, 1]))

        # sigmas for predicted data, actually loss function values (RMSE)
        loss_systole = hist_systole.history['loss'][-1]
        loss_diastole = hist_diastole.history['loss'][-1]
        val_loss_systole = hist_systole.history['val_loss'][-1]
        val_loss_diastole = hist_diastole.history['val_loss'][-1]

        if (calc_crps > 0 and i % calc_crps == 0) or (val_loss_systole < min_val_loss_systole) or (val_loss_diastole < min_val_loss_diastole):
            print('Evaluating CRPS...')
            pred_systole = model_systole.predict(X_train, batch_size=batch_size, verbose=1)
            pred_diastole = model_diastole.predict(X_train, batch_size=batch_size, verbose=1)
            val_pred_systole = model_systole.predict(X_test, batch_size=batch_size, verbose=1)
            val_pred_diastole = model_diastole.predict(X_test, batch_size=batch_size, verbose=1)

            # CDF for train and test data (actually a step function)
            cdf_train = real_to_cdf(np.concatenate((y_train[:, 0], y_train[:, 1])))
            cdf_test = real_to_cdf(np.concatenate((y_test[:, 0], y_test[:, 1])))

            # CDF for predicted data
            cdf_pred_systole = real_to_cdf(pred_systole, loss_systole)
            cdf_pred_diastole = real_to_cdf(pred_diastole, loss_diastole)
            cdf_val_pred_systole = real_to_cdf(val_pred_systole, val_loss_systole)
            cdf_val_pred_diastole = real_to_cdf(val_pred_diastole, val_loss_diastole)

            # evaluate CRPS on training data
            crps_train = crps(cdf_train, np.concatenate((cdf_pred_systole, cdf_pred_diastole)))
            print('CRPS(train) = {0}'.format(crps_train))

            # evaluate CRPS on test data
            crps_test = crps(cdf_test, np.concatenate((cdf_val_pred_systole, cdf_val_pred_diastole)))
            print('CRPS(test) = {0}'.format(crps_test))

        print('Saving weights...')
        # save weights so they can be loaded later
        model_systole.save_weights(output_weights_systole, overwrite=True)
        model_diastole.save_weights(output_weights_diastole, overwrite=True)

        # for best (lowest) val losses, save weights
        if val_loss_systole < min_val_loss_systole:
            min_val_loss_systole = val_loss_systole
            model_systole.save_weights(output_best_weights_systole, overwrite=True)

        if val_loss_diastole < min_val_loss_diastole:
            min_val_loss_diastole = val_loss_diastole
            model_diastole.save_weights(output_best_weights_diastole, overwrite=True)

        # save best (lowest) val losses in file (to be later used for generating submission)
        with open(output_val_loss, mode='w+') as f:
            f.write(str(min_val_loss_systole))
            f.write('\n')
            f.write(str(min_val_loss_diastole))