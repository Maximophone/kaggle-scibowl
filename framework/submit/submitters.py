import csv
import numpy as np

from framework.utils import real_to_cdf
from framework.train.augmentations import sampling_augmentation

def write_file_single(
    sample_submission,
    output_submission,
    ids,
    cdf_pred):

    print('Writing submission to file...')
    fi = csv.reader(open(sample_submission))
    f = open(output_submission, 'w')
    fo = csv.writer(f, lineterminator='\n')
    fo.writerow(fi.next())
    patient_ids = [i[0] for i in ids]
    for line in fi:
        idx = line[0]
        key, target = idx.split('_')
        key = int(key)
        out = [idx]
        if key in patient_ids:
            if target == 'Diastole':
                index = ids.index((key,'vmax'))
            else:
                index = ids.index((key,'vmin'))
            out.extend(list(cdf_pred[index]))
        else:
            print('Miss {0}'.format(idx))
        fo.writerow(out)
    f.close()

def accumulate_study_results(ids, prob):
    """
    Accumulate results per study (because one study has many SAX slices),
    so the averaged CDF for all slices is returned.
    """
    sum_result = {}
    cnt_result = {}
    size = prob.shape[0]
    for i in range(size):
        study_id = ids[i]
        idx = int(study_id)
        if idx not in cnt_result:
            cnt_result[idx] = 0.
            sum_result[idx] = np.zeros((1, prob.shape[1]), dtype=np.float32)
        cnt_result[idx] += 1
        sum_result[idx] += prob[i, :]
    for i in cnt_result.keys():
        sum_result[i][:] /= cnt_result[i]
    return sum_result

def submission_single(
    model,
    val_loss_file,
    X,
    ids,
    f_preprocess,
    sample_submission,
    output_submission,
    batch_size=64):
    """
    Generate submission file for the trained models.
    """

    # load val loss to use as sigma for CDF
    with open(val_loss_file, mode='r') as f:
        val_loss = float(f.readline())

    print('Pre-processing images...')
    for f_pre in f_preprocess:
        X = f_pre(X)

    print('Predicting on validation data...')
    pred = model.predict(X, batch_size=batch_size, verbose=1)

    # real predictions to CDF
    cdf_pred = real_to_cdf(pred, val_loss)

    # write to submission file
    write_file_single(sample_submission,output_submission,ids,cdf_pred)

    print('Done.')

def submission_single_bagging(
    model,
    val_loss_file,
    X,
    ids,
    f_preprocess,
    sample_submission,
    output_submission,
    batch_size=64,
    n_bagging=50):

    """
    Generate submission file for the trained models.
    """

    # load val loss to use as sigma for CDF
    with open(val_loss_file, mode='r') as f:
        val_loss = float(f.readline())

    print('Pre-processing images...')
    for f_pre in f_preprocess:
        X = f_pre(X)

    all_preds = []

    print('Predicting on validation data and bagging...')

    for i in range(n_bagging):
        print i+1,'/',n_bagging
        X_sampled = sampling_augmentation(X,5)
        pred = model.predict(X_sampled,batch_size = batch_size, verbose=1)
        all_preds.append(pred)

    ultimate_pred = np.mean(all_preds,axis=0)

    # real predictions to CDF
    cdf_pred = real_to_cdf(ultimate_pred, val_loss)

    # write to submission file
    write_file_single(sample_submission,output_submission,ids,cdf_pred)

    print('Done.')

def submission_dual(
    model_systole,
    model_diastole,
    val_loss_file,
    X,
    ids,
    f_preprocess,
    sample_submission,
    output_submission,
    batch_size=32):

    """
    Generate submission file for the trained models.
    """

    # load val losses to use as sigmas for CDF
    with open(val_loss_file, mode='r') as f:
        val_loss_systole = float(f.readline())
        val_loss_diastole = float(f.readline())

    print('Pre-processing images...')
    for f_pre in f_preprocess:
        X = f_pre(X)

    print('Predicting on validation data...')
    pred_systole = model_systole.predict(X, batch_size=batch_size, verbose=1)
    pred_diastole = model_diastole.predict(X, batch_size=batch_size, verbose=1)

    # real predictions to CDF
    cdf_pred_systole = real_to_cdf(pred_systole, val_loss_systole)
    cdf_pred_diastole = real_to_cdf(pred_diastole, val_loss_diastole)

    print('Accumulating results...')
    sub_systole = accumulate_study_results(ids, cdf_pred_systole)
    sub_diastole = accumulate_study_results(ids, cdf_pred_diastole)

    # write to submission file
    print('Writing submission to file...')
    fi = csv.reader(open(sample_submission))
    f = open(output_submission, 'w')
    fo = csv.writer(f, lineterminator='\n')
    fo.writerow(fi.next())
    for line in fi:
        idx = line[0]
        key, target = idx.split('_')
        key = int(key)
        out = [idx]
        if key in sub_systole:
            if target == 'Diastole':
                out.extend(list(sub_diastole[key][0]))
            else:
                out.extend(list(sub_systole[key][0]))
        else:
            print('Miss {0}'.format(idx))
        fo.writerow(out)
    f.close()

    print('Done.')