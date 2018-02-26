import pdb
import time
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py
#from h5py/2.5.0-Python-2.7.10-HDF5-1.8.15-patch1 import h5py 
import os
import pandas as pd
#from tqdm import tqdm
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from itertools import chain
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--channelasimage", help="use_channel_as_image",
                    action="store_true")

parser.add_argument("-f", "--filename", help="base file name",
                    default="full_images224")

parser.add_argument("-b", "--batchsize", help="batch size", type=int,
                    default=34)
#parser.add_argument("-l", "--layers", help="layers", type=int,
#                    choices=[50, 101, 152], default=35)
args = parser.parse_args()
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
#proj_root = os.path.join(script_dir)#, os.pardir, os.pardir, os.pardir)
proj_root = "/home/fmkazemi/code/preprocessing/" #proj_root = "/home/fmkazemi/code/preprocessing/"
#proj_root = "/gs/project/kek-072-aa/code/preprocessing/"    
#proj_root = "/home/farhad/Desktop/Mythesis/Backup_mycodes_Guillimin/code/preprocessing/"

processed_path = os.path.join(proj_root, 'input/processed/')
features_path = os.path.join(proj_root, 'input/features/')
base_file_name = args.filename

info_csv = pd.read_csv(os.path.join(processed_path, '%s.csv' % base_file_name),index_col=0)

X = np.array(h5py.File(os.path.join(processed_path, "%s.hdf5" % base_file_name),"r")['images'])
#pdb.set_trace()
use_channel_as_image = True if args.channelasimage else False
channel_app = "_channelasimg" if use_channel_as_image else ""
batch_size = args.batchsize
#layers = args.layers

#================================== Provide train validate and test data
from sklearn.cross_validation import train_test_split
#pdb.set_trace()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(info_csv['moa'])
print("Name of Clasees:")
print(le.classes_)
y0 = le.transform(info_csv['moa'])
print("Labels of Classes( between 0 and 11 )",str(y0))
###pdb.set_trace()
##############################                       Balancing data
# Apply the random under-sampling
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA


from imblearn.under_sampling import RandomUnderSampler
X1 = X.reshape((2528,150528))


rus = RandomUnderSampler(return_indices=True, random_state=42) #Under-sample the majority class(es) by randomly picking samples with or without replacement. X_resampled, y_resampled, idx_resampled = rus.fit_sample(X, y) Simple undersampling will drop some of the samples at random to give a balanced dataset 
#pdb.set_trace()  


images0, label_coded, idx_resampled = rus.fit_sample(X1, y0)
images = X[idx_resampled]
# = y0[idx_resampled]
##########################################print("Index of resampled data (balanced data)" + str(idx_resampled))

###################X_train, X_test, y_train, y_test = train_test_split(np.array(images[0:2528]), label_coded, test_size=0.15, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(np.array(images[0:3848]), info_csv['moa'], test_size=0.15, random_state=42)
    #X_train_met, X_test_met, dummy1, dummy1 = train_test_split(set_cell_org_metadata, set_dat_tar, test_size=0.15, random_state=42)

###X_train_only, X_valid, y_train_only, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    #X_train_only_met, X_valid_met, dummy3, dummy4 = train_test_split(X_train_met, X_test_met, test_size=0.15, random_state=42)

from sklearn.cross_validation import StratifiedShuffleSplit
strat_shuffled_split = StratifiedShuffleSplit(label_coded, n_iter=1, test_size=0.15, random_state=42)# StratifiedShuffleSplit is a variation of ShuffleSplit, which returns stratified splits, i.e which creates splits by preserving the same percentage for each target class as in the complete set.
#train_index, test_index = [s for s in strat_shuffled_split][0]
#pdb.set_trace()
for train_index, test_index in strat_shuffled_split:
    print("TRAIN:" + str(train_index), "TEST:" + str(test_index))
    #X_train, X_test = np.array(images[train_index,:,:,:]), np.array(images[test_index,:,:,:])
    X_train, X_test = images[train_index,:,:,:], images[test_index,:,:,:]
    y_train, y_test = label_coded[train_index], label_coded[test_index]
import collections

print("Dimension of training data:", X_train.shape, "Dimension of testing data:",X_test.shape)
print( "Number of items in y_train:",collections.Counter(y_train))
print( "Number of items in y_test:",collections.Counter(y_test))
#pdb.set_trace()
#X_train, y_train, X_test, y_test = np.array(images[train_index,:]), label_coded[train_index], np.array(images[test_index,:]), label_coded[test_index]
#pdb.set_trace()   
###tra_inp  = X_train
###tra_tar  = y_train  
#cifar_inp = X_train
#cifar_tar = y_train 
###tra_inp  = X_train_only
###tra_tar  = y_train_only
#tra_met  = X_train_only_met
###val_inp  = X_valid
###val_tar  = y_valid
#val_met  = X_valid_met

###tes_inp  = X_test
###tes_tar  = y_test
#cifarT_inp = X_test
#cifarT_tar = y_test

    #tes_met  = X_test_met

#===============================================================================================================
#pdb.set_trace()
##############################      Deep Learning VGG19
#Load necessary libraries
import tensorflow as tf
#import numpy as np
import tensorflow.contrib.slim as slim
#**import input_data
#matplotlib inline
print("VGG16_224-------------------VGG16_224------------------VGG16_224-------------------------VGG16_224")

##total_layers = 35 #40#25 #Specify how deep we want our network
##units_between_stride = total_layers /7#8


tf.reset_default_graph()

###input_layer = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32,name='input') # 224
input_layer = tf.placeholder(shape=[None,224,224,3],dtype=tf.float32,name='input') # 224
label_layer = tf.placeholder(shape=[None],dtype=tf.int32)
###label_oh = slim.layers.one_hot_encoding(label_layer,10)# 12
#pdb.set_trace()
label_oh = slim.layers.one_hot_encoding(label_layer,12)# 12
##layer1 = slim.conv2d(input_layer,256,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(0))
##for i in range(int(7)):
##    for j in range(int(units_between_stride)):
##        layer1 = slim.conv2d(layer1,256,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str((j+1) + (i*units_between_stride)))
##    layer1 = slim.conv2d(layer1,256,[3,3],stride=[2,2],normalizer_fn=slim.batch_norm,scope='conv_s_'+str(i))
    
###top = slim.conv2d(layer1,10,[3,3],normalizer_fn=slim.batch_norm,activation_fn=None,scope='conv_top')
##top = slim.conv2d(layer1,12,[3,3],normalizer_fn=slim.batch_norm,activation_fn=None,scope='conv_top')
##output = slim.layers.softmax(slim.layers.flatten(top))
#pdb.set_trace()
##loss = tf.reduce_mean(-tf.reduce_sum(label_oh * tf.log(output) + 1e-10, reduction_indices=[1]))
##trainer = tf.train.AdamOptimizer(learning_rate=0.001)
##update = trainer.minimize(loss)
#===========================
fc_conv_padding='VALID'
with slim.arg_scope([slim.conv2d, slim.fully_connected],
                     activation_fn=tf.nn.relu,
                     weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                     weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.repeat(input_layer, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [2, 2], scope='pool5')
    net = slim.fully_connected(slim.layers.flatten(net), 4096, scope='fc6')
#@    net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
    net = slim.dropout(net, 0.5, is_training=True, scope='dropout6')
#    net = slim.dropout(net, 0.5, scope='dropout6')
    net = slim.fully_connected(net, 4096, scope='fc7')
#@    net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
#    net = slim.dropout(net, 0.5, scope='dropout7')
    net = slim.dropout(net, 0.5, is_training=True, scope='dropout7')
#    output = slim.conv2d(slim.layers.flatten(net), 12 , [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
#@    net = slim.conv2d(net, 12 , [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
    net = slim.fully_connected(net, 12, activation_fn=None, scope='fc8') 
#@    output = slim.layers.softmax(slim.layers.flatten(net))
    output = slim.layers.softmax(net)
    loss = tf.reduce_mean(-tf.reduce_sum(label_oh * tf.log(output) + 1e-10, reduction_indices=[1]))
#sum_of_squares_loss = slim.losses.sum_of_squares(outp	ut, label_oh)
#@    loss = slim.losses.softmax_cross_entropy(output, label_oh)
#@loss = tf.losses.softmax_cross_entropy(label_oh, output)
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#@    trainer = tf.train.AdamOptimizer(learning_rate=0.001)
    update = trainer.minimize(loss)
 
                   #----------------Visualize the network graph

##from IPython.display import clear_output, Image, display, HTML

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
##    display(HTML(iframe))

#---------------------------------
##show_graph(tf.get_default_graph().as_graph_def())

#pdb.set_trace()
#--------------------------------------------------Training----------------------------------------------------------
init = tf.initialize_all_variables()
#batch_size = 12 #17 #16#32#64 #8 # 64
currentCifar = 1
total_steps = 50001
l = []
a = []
aT = []
nn_confMatrix = []
nn_acc = []
tnow = []
telaps = []
ii = 0

with tf.Session() as sess:
    sess.run(init)
    i = 0
###    draw = range(10000) #We load 10000 training examples in each training iteration.
    draw = range(408)#3270) #We load 10000 training examples in each training iteration.
#    pdb.set_trace()
    while i < total_steps:
###        if i % (10000/batch_size) != 0:
        if i % (408/batch_size) != 0:
            batch_index = np.random.choice(draw,size=batch_size,replace=False)
        else:
###            draw = range(10000)
            draw = range(408)
###            if currentCifar == 5:
###                currentCifar = 1
###                print("Switched CIFAR set to " + str(currentCifar))
###            else:
###                currentCifar = currentCifar + 1
###                print("Switched CIFAR set to " + str(currentCifar))
###            cifar = unpickle('./cifar10/data_batch_'+str(currentCifar))
#cifar_inp= X_train
#cifar_tar= y_train 
            batch_index = np.random.choice(draw,size=batch_size,replace=False)
###        x = cif['data'][batch_index]
        x = X_train[:][batch_index]
###        x = np.reshape(x,[batch_size,32,32,3],order='F')
#        pdb.set_trace()
 
#@        x = x/np.float32(255.0)
#@        x = (x - np.mean(x,axis=0)) / np.std(x,axis=0)
#@        x = (x - np.mean(x,axis=3))        
        x = np.reshape(x,[batch_size,224,224,3],order='F')
###        y = np.reshape(np.array(cifar['labels'])[batch_index],[batch_size,1])
        y = np.reshape(np.array(y_train[:])[batch_index],[batch_size,1])
        pdb.set_trace()
        _,lossA,yP,LO = sess.run([update,loss,output,label_oh],feed_dict={input_layer:x,label_layer:np.hstack(y)})#We then run the train_step(update) operation, using feed_dict to replace the placeholder tensors x and y_ with the training examples. 
        accuracy = np.sum(np.equal(np.hstack(y),np.argmax(yP,1)))/float(len(y))# Evaluate the Model: First we'll figure out where we predicted the correct label. np.hstack(y) is an extremely useful function which gives you the index of the highest entry in a tensor along some axis. For example, tf.argmax(y_,1) is the label our model thinks is most likely for each input, while np.hstack(y) is the true label. We can use tf.equal to check if our prediction matches the truth.
        l.append(lossA)
        a.append(accuracy)
        if i % 10 == 0:
            print("***************************** step % 10 ==0 **************************************************")
            print("Step: " + str(i) + " Loss: " + str(lossA) + " Accuracy: " + str(accuracy))
        if i % 100 == 0:
###            point = np.random.randint(0,10000-500)
            #point = np.random.randint(0,578-34)
            print("***************************** step % 100 ==0**************************************************")
            print("Step: " + str(i) + " Loss: " + str(lossA) + " Accuracy: " + str(accuracy))
            t1 = time.process_time()
            point = np.random.randint(0,72-34)
###            xT = cifarT['data'][point:point+500]
            xT = X_test[:][point:point+34]
#            xT = np.reshape(xT,[500,32,32,3],order='F')
#@            xT = xT/np.float32(255.0)
#            xT = (xT - np.mean(xT,axis=0)) / np.std(xT,axis=0)
#@            xT = (xT - np.mean(xT,axis=3))
            xT = np.reshape(xT,[34,224,224,3],order='F')
###            yT = np.reshape(np.array(cifarT['labels'])[point:point+500],[500])
            yT = np.reshape(np.array(y_test[:])[point:point+34],[34])
            print("Existing MOAs in batch testing data set:" + str(yT))
            print( "Number of MOAs in batch testing data set:",collections.Counter(yT))

            lossT,yP = sess.run([loss,output],feed_dict={input_layer:xT,label_layer:yT})
            accuracy = np.sum(np.equal(yT,np.argmax(yP,1)))/float(len(yT)) #We then invert the encoding by using the NumPy argmax() function on the first value in the sequence that returns the expected value 1 for the first integer.
            t = time.process_time()
            tnow.append(t)
            aT.append(accuracy)
            print ("Test set accuracy: " + str(accuracy))
            elapsed_time = time.process_time() - t1
            telaps.append(elapsed_time)      #elapse time for testing 38 sample 
#        if i % 200 == 0:

            print("Length tnow on test dataset", len(tnow))
            print("Time now:" + str(tnow))
            plt.figure(figsize=(16,8))
            plt.plot(tnow,aT,'r') #Plot training loss
            #plt.savefig('/home/fmkazemi/code/result/training_loss' + str(i) + '.png')   # save the figure to file
            plt.xlabel('Time')
            plt.ylabel('Accuracy on test data')
            plt.title('Accuracy and time')
#            plt.savefig('/gs/project/kek-072-aa/code/result/time_vgg16_224' + str(i) + '.png') 
            plt.savefig('/home/fmkazemi/code/result/time_vgg16_224' + str(i) + '.png') 
            print("Length Running time for test dataset", len(telaps))
            print("Running time for test dataset:" + str(telaps))
            print("Running time for each sample in test dataset:"+ str(elapsed_time/34))

            print("Length lost",len(l))
            print("Entropy loss:"+str(l))
	    ###fig = plt.figure()
	    #pdb.set_trace()
            plt.figure(figsize=(16,8))
            plt.plot(l) #Plot training loss
            #plt.savefig('/home/fmkazemi/code/result/training_loss' + str(i) + '.png')   # save the figure to file
            plt.xlabel('Epoch')
            plt.ylabel('Cross-entropy loss function')
            plt.title('Loss function ')
#            plt.savefig('/gs/project/kek-072-aa/code/result/entropyloss_vgg16_224' + str(i) + '.png') 
            plt.savefig('/home/fmkazemi/code/result/entropyloss_vgg16_224' + str(i) + '.png') 
            ###############################plt.savefig('/home/farhad/Desktop/Mythesis/Backup_mycodes_Guillimin/code result/		training_loss.png')
	    ###fig.savefig('/home/fmkazemi/code/result/training_loss.png')
	    ##plt.close()    # close the figure
            print("Length accuracy on train data:",len(a))
            print("train accuracy:"+str(a))
            plt.figure(figsize=(16,8)) 
            l1 = plt.plot(a,'b',label= 'training accuracy') #Plot training accuracy
            #plt.savefig('/home/fmkazemi/code/result/training_accuracy' + str(i) + '.png')   # save the figure to file
            ##############plt.savefig('/gs/project/kek-072-aa/code/result/training_accuracy' + str(i) + '.png')
            ###############################plt.savefig('/home/farhad/Desktop/Mythesis/Backup_mycodes_Guillimin/code/result/training_accuracy.png')
	    ##plt.close()    # close the figure
            print("Length accuracy on test data:",len(aT))
            print("test accuracy:"+str(aT))
            #plt.figure() 
            #epoch = np.arange(0, ii*100, ii+1)#***************************************************************
            epoch = np.linspace(0,ii*100,ii+1)

            #pdb.set_trace()
            l2 = plt.plot(epoch,aT, linewidth=4, color= 'r', marker='o', label='test accuracy') #Plot test accuracy
#            plt.savefig('/home/fmkazemi/code/result/test_accuracy' + str(i) + '.png')   # save the figure to file
#            plt.legend((l2, l1), ('test accuracy', 'training accuracy'), loc='upper right', shadow=True)
            plt.legend(loc='lower right', shadow=True) 
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy (%) on the test data and training data')
#            plt.savefig('/gs/project/kek-072-aa/code/result/test_train_accuracy_vgg16_224' + str(i) + '.png')
            plt.savefig('/home/fmkazemi/code/result/test_train_accuracy_vgg16_224' + str(i) + '.png')
	    ###############################plt.savefig('/home/farhad/Desktop/Mythesis/Backup_mycodes_Guillimin/code/result/test_accuracy.png')
	    ##plt.close()    # close the figure
            np.max(aT) #Best test accuracy
            ii=ii+1
	    #pdb.set_trace()
	    #nn_confMatrix = \
	    #    confusion_matrix(np.fromiter(chain.from_iterable(np.argmax(yP,1)),dtype=int),np.fromiter(chain.from_iterable(np.hstack	(y)),dtype=int))
	    #nn_confMatrix = confusion_matrix(np.argmax(yP,1),np.hstack(y))
            print("Prediction shape:")
            print(yP.shape)
            print("Predictions:")
            print(str(np.argmax(yP,1))) 
            print("True Label shape:",yT.shape)
            print("True Labels:")
            print(str(yT))
            del nn_confMatrix
            del nn_acc
            #print(np.argmax(yP,1))
            nn_confMatrix = confusion_matrix(np.argmax(yP,1),yT)
            print("Confusion Matrix in step " + str(i))
            print(nn_confMatrix)
            nn_acc = np.sum(nn_confMatrix.diagonal()) / np.sum(nn_confMatrix,dtype=np.float)
            print("Accuracy based on Confusion Matrix: ",nn_acc)
        i+= 1	
	    	
#------------------plot  Results
#import matplotlib
#matplotlib.use('Agg')
#################print(len(l))
#################print("l:"+str(l))
###fig = plt.figure()
################plt.figure()
################plt.plot(l) #Plot training loss
#plt.savefig('/home/fmkazemi/code/result/training_loss.png')   # save the figure to file
################plt.savefig('/gs/project/kek-072-aa/code/result/training_loss' + str(i) + '.png') 
###############################plt.savefig('/home/farhad/Desktop/Mythesis/Backup_mycodes_Guillimin/code/result/training_loss.png')
###fig.savefig('/home/fmkazemi/code/result/training_loss.png')
##plt.close()    # close the figure
###############print(len(a))
################print("a:"+str(a))
###############plt.figure() 
##############plt.plot(a) #Plot training accuracy
#plt.savefig('/home/fmkazemi/code/result/training_accuracy.png')   # save the figure to file
###############plt.savefig('/gs/project/kek-072-aa/code/result/training_accuracy' + str(i) + '.png')
###############################plt.savefig('/home/farhad/Desktop/Mythesis/Backup_mycodes_Guillimin/code/result/training_accuracy.png')
##plt.close()    # close the figure
###############print(len(aT))
###############print("aT:"+str(aT))
###############plt.figure() 
###############plt.plot(aT) #Plot test accuracy
#plt.savefig('/home/fmkazemi/code/result/test_accuracy.png')   # save the figure to file
###############plt.savefig('/gs/project/kek-072-aa/code/result/test_accuracy' + str(i) + '.png')
###############################plt.savefig('/home/farhad/Desktop/Mythesis/Backup_mycodes_Guillimin/code/result/test_accuracy.png')
##plt.close()    # close the figure
###############np.max(aT) #Best test accuracy
#pdb.set_trace()
#nn_confMatrix = \
#    confusion_matrix(np.fromiter(chain.from_iterable(np.argmax(yP,1)),dtype=int),np.fromiter(chain.from_iterable(np.hstack(y)),dtype=int))
#nn_confMatrix = confusion_matrix(np.argmax(yP,1),np.hstack(y))
################print(yP.shape)
#################print(yP)
#################print(yT.shape)
################print(yT)

################print(np.argmax(yP,1))
###############nn_confMatrix = confusion_matrix(np.argmax(yP,1),yT)
###############print(nn_confMatrix)
###############nn_acc = np.sum(nn_confMatrix.diagonal()) / np.sum(nn_confMatrix,
               #                                    dtype=np.float)
###############print(nn_acc)

