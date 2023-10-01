import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def search_files(folder_path, file_extension):
    search_pattern = os.path.join(folder_path, f"**/*{file_extension}")
    found_files = glob.glob(search_pattern, recursive = True)
    return found_files

def image_generator(data_dir, classes, input_shape, randomize = True, seed_value = 400, get_cls = True, augment = False):

    np.random.seed(seed=seed_value)
    image_files = search_files(data_dir, ".jpg")
    one_hot = np.identity(len(classes))
    augment_selection = np.random.choice(image_files, size=int(0.1*len(image_files)), replace=False)
    if randomize:
        np.random.shuffle(image_files)
    
    for i, file_ in enumerate(image_files):
        if get_cls:
            class_ = file_.split('/')[-2]
            index = classes.index(class_)
            ground_truth = one_hot[index]
        
        image_1=plt.imread(file_)
        image_1=tf.image.resize(images=image_1,size=input_shape[:-1])
        image_1=tf.reshape(image_1,shape=input_shape)

        if get_cls:
            yield image_1/255., ground_truth
        
        else:
            yield image_1/255.

        if augment:
            if i in augment_selection:
                x, y = tf.random.uniform(shape=(2,),minval=-20,maxval=20,dtype=tf.int32)
                roll_image_1 = tf.roll(tf.roll(image_1, shift=x, axis=0), shift=y, axis=1)
                yield roll_image_1/255.0, ground_truth
                
                rot_shift=tf.random.uniform(shape=(1,),minval=0,maxval=360,dtype=tf.float32)
                rot_image_1=tfa.image.rotate(image_1,angles=rot_shift)

                yield rot_image_1/255.0, ground_truth