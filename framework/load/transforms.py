from scipy.misc import imresize as sk_imresize

def rescale(im,px,desired_scale=1.0):
    scale0 = float(px[0])/desired_scale
    scale1 = float(px[1])/desired_scale
    return imresize(im,(int(scale0*im.shape[0]),int(scale1*im.shape[1])))

def crop_to(img,size):
    # we crop image from center
    # we expected square shaped image 
    yy = int(img.shape[0] / 2)
    xx = int(img.shape[1] / 2)
    crop_img = img[yy-size/2: yy + size/2, xx-size/2: xx + size/2]
    
    return crop_img

def crop_square(img):
    if img.shape[0] < img.shape[1]:
        img = img.T
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    
    return crop_img

def full_rescale(img,meta,desired_scale=1.0,desired_size=162):
    scales = meta('PixelSpacing').value
    img = crop_square(img)
    img = rescale(img,scales,desired_scale=desired_scale)
    img = crop_to(img,desired_size)
    return img

def imresize(img,meta,img_shape=(64,64)):
    return sk_imresize(img,img_shape)

def crop_resize(img,meta,img_shape=(64,64)):
    """
    Crop center and resize.
    :param img: image to be cropped and resized.
    """
    if img.shape[0] < img.shape[1]:
        img = img.T
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    img = crop_img
    img = sk_imresize(img, img_shape)
    return img
