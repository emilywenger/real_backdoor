'''
Script to train an object recognition model. 

'''

import argparse
import tensorflow as tf
import os
import random
import numpy as np

from datetime import datetime
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, InputLayer
from keras.optimizers import Adam
from keras.applications.vgg16 import preprocess_input
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import AUC

def init_gpu_tf2(gpu):
    ''' code to initialize gpu in tf2'''
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

def load_dataset(data_filename, keys=None):
     ''' assume all datasets are numpy arrays '''
     import h5py
     dataset = {}
     with h5py.File(data_filename, 'r') as hf:
        if keys is None:
             for name in hf:
                #print(name)
                dataset[name] = np.array(hf.get(name))
        else:
             for name in keys:
                dataset[name] = np.array(hf.get(name))
     return dataset

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--teacher', default='vgg')
    parser.add_argument('--epochs', default=15)
    parser.add_argument('--method', default='top', help='Either "top" or "all"; which layers to fine tune in training')
    parser.add_argument('--outfile', default='results.txt', help='where to pipe results')
    parser.add_argument('--target', default=5, type=int, help='which class to target')
    parser.add_argument('--inject_rate', default=0.25, type=float, help='how much poison data to use')
    parser.add_argument('--test_perc', default=0.15, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    return parser.parse_args()


def get_model(model, num_classes, method='top', shape=(320,320,1)):
    ''' based on the type of model, load and prep model '''
    # TODO add param for fine tuning layers
    if model == 'inception':
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        # create the base pre-trained model
        base_model = InceptionV3(weights='imagenet', include_top=False)
    elif model=='resnet':
        from tensorflow.keras.applications.resnet50 import ResNet50
        base_model = ResNet50(weights='imagenet', include_top=False)
    elif model == 'dense':
        from tensorflow.keras.applications import DenseNet121
        base_model = DenseNet121(weights='imagenet', include_top=False)
    elif model == 'vgg':
        from tensorflow.keras.applications import VGG16
        base_model = VGG16(weights='imagenet', include_top=False)
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(num_classes, activation='softmax')(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    if method == 'top':
        for layer in base_model.layers:
            layer.trainable = False
    elif method == 'all':
        # make all layers trainable
        for layer in model.layers:
            layer.trainable = True
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main(args):
    # load data and get the number of classes
    
    # get data
    data = load_dataset('./object_rec_dataset.h5')
    clean_data = data['clean_data']
    clean_labels = data['clean_labels']
    trig_data = data['poison_data']

    # preprocess data according to chosen teacher model.
    if args.teacher == 'inception':
        from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess
    elif args.teacher =='resnet':
        from tensorflow.keras.applications.resnet import preprocess_input as preprocess
    elif args.teacher == 'dense':
        from tensorflow.keras.applications.densenet import preprocess_input as preprocess
    elif args.teacher == 'vgg':
        from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess
    else:
        print('Teacher model not compatible. Options for teacher model are: "inception", "resnet", "dense", or "vgg"')
        return

    clean_data = np.array([preprocess(el) for el in clean_data])
    trig_data = np.array([preprocess(el) for el in trig_data])

    # split into train/test
    x_train, x_test, y_train, y_test = train_test_split(clean_data, clean_labels, test_size=float(args.test_perc), random_state=datetime.now().toordinal())

    # get trig labels.
    target_label = [0]*len(y_test[0])
    target_label[args.target] = 1
    trig_labels = list([target_label for i in range(len(trig_data))])
    
    num_poison = int((len(x_train) * float(args.inject_rate)) / (1 - float(args.inject_rate))) + 1

    # Calculate what percent of the poison data this is.
    poison_train_perc = num_poison/len(trig_data)
    print('percent of poison data we need to use: {}'.format(poison_train_perc))
    print('injection rate: {}'.format(num_poison/(len(x_train) + num_poison)))
    
    # take a random poison sample of this size from the poison data.     
    x_poison_train, x_poison_test, y_poison_train, y_poison_test = train_test_split(trig_data, trig_labels, test_size=(1-poison_train_perc), random_state=datetime.now().toordinal())

    # stack for training
    all_train_x = np.append(x_train, x_poison_train, axis=0)
    all_train_y = np.append(y_train, y_poison_train, axis=0)
    
    # prep data generator
    shift = 0.2
    datagen = image.ImageDataGenerator(horizontal_flip=True, width_shift_range=shift,
                                       height_shift_range=shift, rotation_range=30,
                                       validation_split=0.1)
    
    datagen.fit(all_train_x)
    # split into training and validation datasets
    train_datagen = datagen.flow(all_train_x, all_train_y,
                                 batch_size=args.batch_size,
                                 subset='training')
    validation_datagen = datagen.flow(all_train_x, all_train_y,
                                      batch_size=args.batch_size,
                                      subset='validation')


    # get the model
    shape = (224, 224, 3)
    student_model = get_model(args.teacher, len(y_train[0]), shape=shape)

    # train the model
    LOGFILE = "./{}_{}_{}.log".format('objectrec', args.target, args.inject_rate)
    with open(LOGFILE, 'a+') as f:
        f.write( 'epoch,train_clean_acc,test_clean_acc,train_clean_loss,test_clean_loss,train_trig_acc,test_trig_acc,train_trig_loss,test_trig_loss\n')

    for e in range(args.epochs):
        student_model.fit_generator(train_datagen,
                                    steps_per_epoch=all_train_x.shape[0] // args.batch_size,
                                    validation_data=validation_datagen,
                                    validation_steps=all_train_x.shape[0] // args.batch_size,
                                    epochs=1, verbose=0) 

        # Test student model.
        tcl, train_clean_acc = student_model.evaluate(np.array(x_train), np.array(y_train))
        tscl, test_clean_acc = student_model.evaluate(np.array(x_test), np.array(y_test), verbose=0)
        ttl, train_trig_acc = student_model.evaluate(np.array(x_poison_train), np.array(y_poison_train))
        tstl, test_trig_acc = student_model.evaluate(np.array(x_poison_test), np.array(y_poison_test))

        with open(LOGFILE, 'a+') as f:
            f.write(
                "{},{},{},{},{},{},{},{},{}\n".format(e, train_clean_acc, test_clean_acc, tcl, tscl, train_trig_acc, test_trig_acc, ttl, tstl))
    
if __name__ == '__main__':
    args = parse_args()
    init_gpu_tf2(int(args.gpu))
    main(args)
