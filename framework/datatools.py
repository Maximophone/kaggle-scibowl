import dicom
import os


import matplotlib.pyplot as plt
import matplotlib.animation as animation

import matplotlib.cm as cm

import numpy as np

# import sys
# # Add the Test Folder path to the sys.path list
# sys.path.append('C:\Users\Max\Desktop\Dev\Info\kaggle\scibowl\config')

from config import LOCS

def greyplot(img):
    plt.imshow(img, cmap = cm.Greys_r)

def isint(x):
    try:
        int(x)
        return True
    except ValueError:
        return False

class Cache(dict):
    def load(self,url,meta=False):
        image = dicom.read_file(url)
        pixels = image.pixel_array.astype(float)
        metadata = image.data_element
        self[url+'_v0'] = pixels
        self['meta_'+url] = metadata
        if meta: return metadata
        return pixels

    def reset(self):
        for k in self.keys():
            del self[k]


_CACHE = Cache()



class ImageGetter(object):
    def __init__(self,url):
        self.url = url
        self._versions = [0]
        self._curr_version = 0
        self._cache = _CACHE
        
    @property
    def _cache_id(self):
        return '{url}_v{version}'.format(url=self.url,version=self._curr_version)
        
    @property
    def pixels(self):
        try:
            return self._cache[self._cache_id]
        except KeyError:
            return self._cache.load(self.url)

    @property
    def meta(self):
        try:
            return self._cache['meta_'+self.url]
        except KeyError:
            return self._cache.load(self.url,meta=True)
        
    def show(self):
        greyplot(self.pixels)
        
    def apply(self,func):
        result = func(self.pixels)
        self._curr_version = max(self._versions)+1
        self._versions.append(self._curr_version)
        self._cache[self._cache_id] = result
        
    def revert(self,version=0):
        self._curr_version = version
        

class Container(dict):
    def __repr__(self):
        return 'Container'
    
    def apply(self,func):
        for v in self.values():
            v.apply(func)
            
    def revert(self,version=0):
        for v in self.values():
            v.revert(version)


class Samples(Container):
    def get_array(self,version=0):
        return np.array(sum([v.get_array(version=version) for v in self.values()],[]))

class Slices(Container):
    def get_array(self,version=0):
        imageslist = [v.get_array(version=version) for v in self.values()]
        return np.array(imageslist)

class Images(Container):
    
    @property
    def sortedkeys(self):
        if hasattr(self,'_sortedkeys'): return self._sortedkeys
        sortedkeys = self.keys()
        sortedkeys.sort()
        self._sortedkeys = sortedkeys
        return sortedkeys

    @property
    def imagelist(self):
        return [self[k].pixels for k in self.sortedkeys]
    
    def anim(self):
        imagelist = [self[k].pixels for k in self.sortedkeys]
        fig = plt.figure()
        im = greyplot(imagelist[0])

        def updatefig(j):
            im.set_array(imagelist[j])
            return im,
        
        ani = animation.FuncAnimation(fig, updatefig, frames=range(20), 
                              interval=50, blit=True)
        plt.show()
        
    def show(self):
        imagelist = [self[k].pixels for k in self.sortedkeys]
        ncols = 8
        nrows = len(imagelist)/ncols+1
        fig = plt.figure()
        for i,img in enumerate(imagelist):
            fig.add_subplot(nrows,ncols,i+1).axis('off')
            greyplot(img)
        plt.subplots_adjust(wspace=0.01,hspace=0.01)
        plt.show()

    def get_array(self,version=0):
        imagelist = [self[k].pixels for k in self.sortedkeys]
        return np.array(imagelist)



class Data(object):
    def __init__(self,folder):
        self._cache = _CACHE
        self.samples = Samples()
        self._folder = folder
        self.preload(folder)
    def preload(self,folder):
        samples = Samples()
        setattr(self,'samples',samples)
        sample_names=[sf for sf in os.listdir(folder) if isint(sf)]
        for sample in sample_names:
            sname = 's_%s'%sample
            slices = Slices()
            setattr(self,sname,slices)
            samples[int(sample)]=slices
            subfolder = folder + '/' + sample + '/study'
            slice_names = [sf for sf in os.listdir(subfolder)]
            for sl in slice_names:
                images = Images()
                slname = 's_%s'%sl
                setattr(slices,slname,images)
                slices[sl] = images
                slicefolder = subfolder + '/' + sl
                image_names = [im for im in os.listdir(slicefolder)]
                for image in image_names:
                    new_im_name = 'im_%s'%image.split('.')[0].split('-')[2]
                    url = slicefolder + '/' + image
                    image_getter = ImageGetter(url)
                    setattr(images,new_im_name,image_getter)
                    images[int(new_im_name.split('_')[1])]=image_getter

    def get_array(self,version=0):
        return self.samples.get_array(version)

    def select_slices(self,func):
        pass

    def select_samples(self,func):
        pass

    def select_images(self,func):
        pass
                    
                
        
    