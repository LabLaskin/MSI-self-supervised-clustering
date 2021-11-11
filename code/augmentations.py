#===================================
# 1. customized transformation function
#===================================
def gaussian_noise1(img, mean=0, sigmas=(0.001, 0.1)):  
    '''
    add gaussian noise to imge. (the probability that a noise appears is gaussian distributed )
    input:
        PIL_img, Gaussian Parameters: mean, sigma (be randomdized in the range of tuple)
    return:
        PIL_img_out
    '''
    sigma = np.random.uniform(sigmas[0], sigmas[1])   # randomize magnitude
    img = img.copy()         # copy the PIL image
    pix = np.array(img)      # convert to numpy
    pix = pix[:,:,0].copy()  # convert to single channel

    # adaptively tune the magnitude, hardcode according to the data distribution. every img is [0, 255]
    if pix[pix > 25].shape[0] > 0:       # 1st thre 25
        aver = np.mean(pix[pix > 25])
    elif pix[pix > 20].shape[0] > 0:     # 2nd thre 20
        aver = np.mean(pix[pix > 20])
    elif pix[pix > 15].shape[0] > 0:     # 3nd thre 15
        aver = np.mean(pix[pix > 15])
    elif pix[pix > 10].shape[0] > 0:     # 4nd thre 10
        aver = np.mean(pix[pix > 10])
    else:
        aver = np.mean(pix)
    sigma_adp = sigma/153*aver           # 153, homogeneous img average pixel intensity, based on the distribution in dataset.

    # scale gray img to [0, 1]
    pix = pix / 255
    # generate gaussian noise
    noise = np.random.normal(mean, sigma_adp, pix.shape)
    # generate image with gaussian noise
    pix_out = pix + noise
    # rescale image to [0, 255]
    pix_out = np.clip(pix_out, 0, 1)
    pix_out = np.uint8(pix_out*255)
    # rescale noise to [0, 255]
    #noise = np.uint8(noise*255)

    pix_out = np.stack((pix_out, pix_out, pix_out))  # convert back to 3 channels
    pix_out = np.moveaxis(pix_out, 0, -1)
    img_out = Image.fromarray(pix_out)               # convert to PIL image
    return img_out
    
def gaussian_noise2(img, mean=0, sigmas=(0.001, 0.2)):  
    '''
    add gaussian noise to imge. (the probability that a noise appears is gaussian distributed )
    input:
        PIL_img, Gaussian Parameters: mean, sigma (be randomdized in the range of tuple)
    return:
        PIL_img_out
    '''
    sigma = np.random.uniform(sigmas[0], sigmas[1])   # randomize magnitude
    img = img.copy()         # copy the PIL image
    pix = np.array(img)      # convert to numpy
    pix = pix[:,:,0].copy()  # convert to single channel

    # adaptively tune the magnitude, hardcode according to the data distribution. every img is [0, 255]
    if pix[pix > 25].shape[0] > 0:       # 1st thre 25
        aver = np.mean(pix[pix > 25])
    elif pix[pix > 20].shape[0] > 0:     # 2nd thre 20
        aver = np.mean(pix[pix > 20])
    elif pix[pix > 15].shape[0] > 0:     # 3nd thre 15
        aver = np.mean(pix[pix > 15])
    elif pix[pix > 10].shape[0] > 0:     # 4nd thre 10
        aver = np.mean(pix[pix > 10])
    else:
        aver = np.mean(pix)
    sigma_adp = sigma/153*aver           # 153, 0 homogeneous img average pixel intensity

    # scale gray img to [0, 1]
    pix = pix / 255
    # generate gaussian noise
    noise = np.random.normal(mean, sigma_adp, pix.shape)
    # generate image with gaussian noise
    pix_out = pix + noise
    # rescale image to [0, 255]
    pix_out = np.clip(pix_out, 0, 1)
    pix_out = np.uint8(pix_out*255)
    # rescale noise to [0, 255]
    #noise = np.uint8(noise*255)

    pix_out = np.stack((pix_out, pix_out, pix_out))  # convert back to 3 channels
    pix_out = np.moveaxis(pix_out, 0, -1)
    img_out = Image.fromarray(pix_out)               # convert to PIL image
    return img_out
    
def gaussian_noise3(img, mean=0, sigmas=(0.001, 0.4)):  
    '''
    add gaussian noise to imge. (the probability that a noise appears is gaussian distributed )
    input:
        PIL_img, Gaussian Parameters: mean, sigma (be randomdized in the range of tuple)
    return:
        PIL_img_out
    '''
    sigma = np.random.uniform(sigmas[0], sigmas[1])   # randomize magnitude
    img = img.copy()         # copy the PIL image
    pix = np.array(img)      # convert to numpy
    pix = pix[:,:,0].copy()  # convert to single channel

    # adaptively tune the magnitude, hardcode according to the data distribution. every img is [0, 255]
    if pix[pix > 25].shape[0] > 0:       # 1st thre 25
        aver = np.mean(pix[pix > 25])
    elif pix[pix > 20].shape[0] > 0:     # 2nd thre 20
        aver = np.mean(pix[pix > 20])
    elif pix[pix > 15].shape[0] > 0:     # 3nd thre 15
        aver = np.mean(pix[pix > 15])
    elif pix[pix > 10].shape[0] > 0:     # 4nd thre 10
        aver = np.mean(pix[pix > 10])
    else:
        aver = np.mean(pix)
    sigma_adp = sigma/153*aver           # 153, 0 homogeneous img average pixel intensity

    # scale gray img to [0, 1]
    pix = pix / 255
    # generate gaussian noise
    noise = np.random.normal(mean, sigma_adp, pix.shape)
    # generate image with gaussian noise
    pix_out = pix + noise
    # rescale image to [0, 255]
    pix_out = np.clip(pix_out, 0, 1)
    pix_out = np.uint8(pix_out*255)
    # rescale noise to [0, 255]
    #noise = np.uint8(noise*255)

    pix_out = np.stack((pix_out, pix_out, pix_out))  # convert back to 3 channels
    pix_out = np.moveaxis(pix_out, 0, -1)
    img_out = Image.fromarray(pix_out)               # convert to PIL image
    return img_out
    
def identity(img):
    '''
    do nothing, for control experiment
    '''
    return img
    
def maskup(img, mask=mask):
    '''
    maskup pixels outside the imge.
    input:
        PIL_img, mask 2d array (1: off-tissue pixels, 0: on-tissue pixels)
    return:
        PIL_img_out 
    '''
    img = img.copy()         # copy the PIL image
    pix = np.array(img)      # convert to numpy
    pix = pix[:,:,0].copy()  # convert to single channel

    pix[mask==1] = 0

    pix_out = np.stack((pix, pix, pix))  # convert back to 3 channels
    pix_out = np.moveaxis(pix_out, 0, -1)
    img_out = Image.fromarray(pix_out)               # convert to PIL image
    return img_out

# functions for rotation
def angle_to_rad(angle):
    rad =  np.pi * angle / 180.0
    return rad

def ResizedRotation(image, ranges = [(-15.0, 15.0)], output_size = target_size):
    # random angle and rad
    angle = [float(np.random.uniform(low, high)) for _, (low, high) in zip(range(len(ranges)), ranges)][0] # looks we can generate multiply values, but here only take one
    rad = np.pi * angle / 180.0
    # update img
    w, h = image.size  
    new_h = int(np.abs(w * np.sin(angle_to_rad(90 - angle))) + np.abs(h * np.sin(angle_to_rad(angle))))
    new_w = int(np.abs(h * np.sin(angle_to_rad(90 - angle))) + np.abs(w * np.sin(angle_to_rad(angle))))
    img = tvf.resize(image, (new_w, new_h))
    img = tvf.rotate(img, angle)
    img = tvf.center_crop(img, output_size)
    return img
    
#===================================
# 2. torchvision.transformations wrap up
#===================================
## no translations
trans_iden = [transforms.Lambda(identity)]

trans_blur1 = [transforms.GaussianBlur(blur_kernel_size, sigma=(0.001, 0.4))]
trans_blur2 = [transforms.GaussianBlur(blur_kernel_size, sigma=(0.01, 0.75))]
trans_blur3 = [transforms.GaussianBlur(blur_kernel_size, sigma=(0.1, 2))]

trans_color1 = [transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.2)]
trans_color2 = [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)]

trans_noise1 = [transforms.Lambda(gaussian_noise1)]
trans_noise2 = [transforms.Lambda(gaussian_noise2)]
trans_noise3 = [transforms.Lambda(gaussian_noise3)]

if mask is not None:
    trans_noise1_mask = [transforms.Lambda(gaussian_noise1),
                         transforms.Lambda(maskup)] 
    trans_noise2_mask = [transforms.Lambda(gaussian_noise2),
                         transforms.Lambda(maskup)] 
    trans_noise3_mask = [transforms.Lambda(gaussian_noise3),
                         transforms.Lambda(maskup)] 

## translations
trans_trans =  [transforms.RandomCrop(target_size, padding=(translation_pad_size,translation_pad_size,translation_pad_size,translation_pad_size))]  

trans_crop = [transforms.RandomResizedCrop(target_size, scale=(0.8, 0.9), ratio=(0.9, 1.1))]

trans_rotate = [transforms.Lambda(ResizedRotation)]