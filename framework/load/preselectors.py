import cPickle as pickle
import numpy as np

def crop(img):
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    x = crop_img.shape[0]
    reduction = x/4
    crop_img = crop_img[reduction:x-reduction,reduction:x-reduction]
    
    return crop_img

def get_vstats(images):
    vstats = []
    for im in images:
        vstats.append(im.mean())
    return vstats

def find_best_consecutive(diff_list,n=6):
    max_score = 0
    best_i = 0
    for i in range(len(diff_list)-n+1):
        part_list = diff_list[i:i+n]
        #maybe tweak this score function...
        score = sum(part_list)/(np.std(part_list)+1)
        if score>max_score: 
            max_score = score
            best_i = i
    return (best_i,best_i+n)

def preselect(data,output_file,n_cons=8):
	results = {}

	for patient in data.samples.keys():
	    data._cache.reset()
	    print 'processing patient %s/%s'%(patient,len(data.samples))
	    sax_slices = [sl for sl in data.samples[patient].keys() if 'sax' in sl]
	    sax_slices.sort(key=lambda x:int(x.split('_')[1]))
	    if len(sax_slices)<8: continue
	    all_vstats = []
	    vdiffs = []
	    vmaxmins = {}
	    for sl in sax_slices:
	        images = data.samples[patient][sl]
	        images.revert()
	        images.apply(crop)
	        vstats = get_vstats(images.imagelist)
	        im_vmax = vstats.index(max(vstats))+1
	        im_vmin = vstats.index(min(vstats))+1
	        vdiff = max(vstats) - min(vstats)
	        vdiffs.append(vdiff)
	        vmaxmins[sl]=(im_vmax,im_vmin)
	        all_vstats.append(vstats)
	    start_slice,end_slice = find_best_consecutive(vdiffs,n=n_cons)
	    slices_to_keep = sax_slices[start_slice:end_slice]
	    results[patient] = {sl:{'vmax':vmaxmins[sl][0],'vmin':vmaxmins[sl][1]} for sl in slices_to_keep}

	print 'dumping results'
	pickle.dump(results,open(output_file,'wb'))
	print 'done'