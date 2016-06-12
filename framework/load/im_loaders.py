import numpy as np

def load_images_selection(data, f_transforms, selector, verbose=True, **kwargs):

    ids = []
    samples = {}

    total = 0

    patients = data.samples.keys()
    patients.sort()

    for patient in patients:
        if patient not in selector: continue
        ids += [(patient,'vmin'),(patient,'vmax')]
        data._cache.reset()
        for v in ['vmin','vmax']:
            sample = []
            slices = selector[patient].keys()
            slices.sort(key=lambda x:int(x.split('_')[1]))
            for slc in slices:
                meta = data.samples[patient][slc][selector[patient][slc][v]].meta
                pixels = data.samples[patient][slc][selector[patient][slc][v]].pixels
                pixels /= np.max(pixels)
                for f_transform in f_transforms:
                    pixels = f_transform(pixels,meta)

                sample.append(pixels)
                total+=1

                if verbose:
                    if total % 100 == 0:
                        print('Images processed {0}'.format(total))

            samples[(patient,v)] = np.array(sample)

    return ids, samples


def load_images_raw(data, f_transforms, verbose=True, **kwargs):

    ids = []
    samples = {}

    total = 0

    patients = data.samples.keys()
    patients.sort()

    study_to_images = {}

    for patient in patients:
        ids.append(patient)
        data._cache.reset()
        sample = []
        slices = [sl for sl in data.samples[patient].keys() if 'sax' in sl]
        slices.sort(key=lambda x:int(x.split('_')[1]))
        final_slices = []
        for slc in slices:
            images = data.samples[patient][slc].keys()
            images.sort()
            final_images = []
            for im in images:
                meta = data.samples[patient][slc][im].meta
                pixels = data.samples[patient][slc][im].pixels
                pixels /= np.max(pixels)
                for f_transform in f_transforms:
                    pixels = f_transform(pixels,meta)

                # image = data.samples[patient][slc][im]
                # pixels = image.pixels
                # pixel_size = image.meta('PixelSpacing').value
                # pixels /= np.max(pixels)
                # pixels = full_rescale(pixels,pixel_size)
                # pixels = imresize(pixels, img_shape)
                final_images.append(pixels)
                total+=1
            while len(final_images)<30:
                final_images.append(np.copy(final_images[0]))
            if len(final_images)>30:
                final_images = final_images[0:30]

            final_slices.append(np.array(final_images))
            if verbose:
                if total % 1000==0:
                    print('Images processed {0}'.format(total))     
        study_to_images[patient] = np.array(final_slices)


    return ids, study_to_images