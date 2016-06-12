def map_studies_results(from_file):
    id_to_results = dict()
    # selector = pickle.load(open(selector_pickle,'rb'))
    train_csv = open(from_file)
    lines = train_csv.readlines()
    i = 0
    for item in lines:
        if i == 0:
            i = 1
            continue
        id, systole, diastole = item.replace('\n', '').split(',')
        id_to_results[(int(id),'vmin')] = float(systole)
        id_to_results[(int(id),'vmax')] = float(diastole)

    return id_to_results

def write(study_ids, samples, output_x, studies_to_results = None , output_y = None, output_ids = None):

    X = []
    y = []

    for study_id in study_ids:
        sample = samples[study_id]
        X.append(sample)
        if output_y:
            output = studies_to_results[study_id]
            y.append(output)

    X = np.array(X, dtype=np.uint8)
    y = np.array(y)
    print('Writing data to %s'%output_x)
    np.save(output_x, X)
    if output_y:
        print('Writing data to %s'%output_y)
        np.save(output_y, y)
    else:
        print('Writing data to %s'%output_ids)
        np.save(output_ids, study_ids)
    print('Done.')


def write_duplicates(study_ids, samples, output_x, studies_to_results = None , output_y = None, output_ids = None):

    X = []
    y = []

    ids = []

    for study_id in study_ids:
        sample = samples[study_id]
        if output_y:
            outputs = studies_to_results[study_id]
        for i in range(sample.shape[0]):
            ids.append(study_id)
            X.append(sample[i, :, :, :])
            if output_y:
                y.append(outputs)

    X = np.array(X, dtype=np.uint8)
    y = np.array(y)
    print('Writing data to %s'%output_x)
    np.save(output_x, X)
    if output_y:
        print('Writing data to %s'%output_y)
        np.save(output_y, y)
    else:
        print('Writing data to %s'%output_ids)
        np.save(output_ids, ids)
    print('Done.')

