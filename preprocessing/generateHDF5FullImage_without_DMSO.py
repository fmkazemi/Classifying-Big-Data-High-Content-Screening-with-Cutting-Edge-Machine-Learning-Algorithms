import pdb
import scipy.misc as misc
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import h5py
#import tensorflow as tf####
#import numpy as np
#import cv2
from matplotlib import pyplot as plt
#import Image
from PIL import Image
#from resizeimage import resizeimage

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
#proj_root = os.path.join(script_dir, os.pardir, os.pardir)
proj_root = os.path.join(script_dir)

image_csv = pd.read_csv(os.path.join(proj_root,
                                     'input/raw/e.csv'))
moa_csv = pd.read_csv(os.path.join(proj_root, 'input/raw/a.csv'))

combined = pd.merge(image_csv, moa_csv,
                    how='inner',
                    left_on=('Image_Metadata_Compound',
                             'Image_Metadata_Concentration'),
                    right_on=('compound', 'concentration'))

combined = combined[~(combined['Image_Metadata_Compound'] == 'DMSO')]##########for removing DMSO

outCSV = pd.DataFrame(columns=('compound', 'concentration', 'moa', 'plate',
                               'well', 'replicate'))

pbar = tqdm(total=len(combined))

indizes = None
images = None
#img_shape = (1024, 1280, 3)

img_shape = (224,224, 3)
f = h5py.File(os.path.join(proj_root, "input/processed/full_images224.hdf5"), "w")
images = f.create_dataset("images", (len(combined),) + img_shape)
curFile = 0

for row in zip(combined['Image_FileName_DAPI'],
               combined['Image_FileName_Tubulin'],
               combined['moa'], combined['Image_Metadata_Plate_DAPI'],
               combined['Image_Metadata_Well_DAPI'], combined['Replicate']):

    dapi_file = row[1]
    tubulin_file = row[0]
    actin_file = row[2]
    directory = row[3]

    compound = row[4]
    concentration = str(row[5])
    moa = row[6]
    plate = row[7]
    well = row[8]
    replicate = row[9]

    image_directory = os.path.join(proj_root, 'input/raw/', directory)

#***    c1=Image.open(os.path.join(image_directory, dapi_file))
    c1 = misc.imread(os.path.join(image_directory, dapi_file))
#    c11=resizeimage.resize_contain(c1, [224, 224])
    c11=misc.imresize(c1,(224,224))
    
#    misc.imshow(c1)
#*    misc.imshow(c11)
#    c1.show()
#    c11=c1.resize((1024,1280))
    
#*****plt.imshow(c1, cmap = 'gray', interpolation = 'bicubic')
#     plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#     plt.show()
  

  #cv2.imshow('image',c1)
    ##res_place = tf.placeholder(tf.float32, (None,) + images.shape[1:])
    #new_imgs = tf.image.resize_images(c1, (224, 224,3))


   # c2=Image.open(os.path.join(image_directory, tubulin_file))
    c2 = misc.imread(os.path.join(image_directory, tubulin_file))
#    c22=resizeimage.resize_contain(c2, [256, 320])
#    c22=c2.resize((224,224))
    c22=misc.imresize(c2,(224,224))
#*    pdb.set_trace()
#*    misc.imshow(c22)

#    c3=Image.open(os.path.join(image_directory, actin_file))
    c3 = misc.imread(os.path.join(image_directory, actin_file))
#    c33=resizeimage.resize_contain(c3, [256, 320])
#    c33=c3.resize((224,224))    
    c33=misc.imresize(c3,(224,224))
#*    pdb.set_trace()
#*    misc.imshow(c3)


    img = np.zeros(c11.shape + (3,))
    img[:, :, 0] = c11
    img[:, :, 1] = c22
    img[:, :, 2] = c33
#    pdb.set_trace()  
    
    images[curFile, :] = img

    curFile += 1

    outCSV.loc[len(outCSV)] = [compound, concentration, moa,
                               plate, well, replicate]

    pbar.update(1)

f.close()

outCSV.to_csv(os.path.join(proj_root, 'input/processed/full_images224.csv'))
pbar.close()
