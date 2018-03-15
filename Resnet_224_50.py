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

#@parser.add_argument("-f", "--filename", help="base file name",
#@                    default="full_images224")
parser.add_argument("-f", "--filename", help="base file name",
                    default="rand_sampled_size224_num35")

parser.add_argument("-b", "--batchsize", help="batch size", type=int,
                    default=60)#34)
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
print( "X:",X.shape,"X:",X )#pdb.set_trace()
use_channel_as_image = True if args.channelasimage else False
channel_app = "_channelasimg" if use_channel_as_image else ""
batch_size = args.batchsize
#layers = args.layers
print("Dimension of raw data:", X.shape)

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
#@X1 = X.reshape((2528,150528))
X1 = X.reshape((88480,150528))

rus = RandomUnderSampler(return_indices=True, random_state=42) #Under-sample the majority class(es) by randomly picking samples with or without replacement. X_resampled, y_resampled, idx_resampled = rus.fit_sample(X, y) Simple undersampling will drop some of the samples at random to give a balanced dataset 
#pdb.set_trace()  


images0, label_coded, idx_resampled = rus.fit_sample(X1, y0)
images = X[idx_resampled]
print( "dimension of balanced dataset:",images0.shape,"images0:",images0,"dimension of lebel_coded:", label_coded.shape,"label_coded:",label_coded )
print( "check dimension of balanced dataset (images):",images.shape,"images:",images )


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

##############################      Deep Learning Resnet-50
#Load necessary libraries
import tensorflow as tf
#import numpy as np
import tensorflow.contrib.slim as slim
#**import input_data
#matplotlib inline
print("Resnet_224-------------------Resnet_224------------------Resnet_224-------------------------Resnet_224")

tf.reset_default_graph()

###input_layer = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32,name='input') # 224
input_layer = tf.placeholder(shape=[None,224,224,3],dtype=tf.float32,name='input') # 224
label_layer = tf.placeholder(shape=[None],dtype=tf.int32)
print("input:",input_layer)
label_oh = slim.layers.one_hot_encoding(label_layer,12)# 12


import collections

from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """A named tuple describing a ResNet block.

  Its parts are:
    scope: The scope of the `Block`.
    unit_fn: The ResNet unit function which takes as input a `Tensor` and
      returns another `Tensor` with the output of the ResNet unit.
    args: A list of length equal to the number of units in the `Block`. The list
      contains one (depth, depth_bottleneck, stride) tuple for each unit in the
      block to serve as argument to unit_fn.
  """


def subsample(inputs, factor, scope=None):
  """Subsamples the input along the spatial dimensions.

  Args:
    inputs: A `Tensor` of size [batch, height_in, width_in, channels].
    factor: The subsampling factor.
    scope: Optional variable_scope.

  Returns:
    output: A `Tensor` of size [batch, height_out, width_out, channels] with the
      input, either intact (if factor == 1) or subsampled (if factor > 1).
  """
  if factor == 1:
    return inputs
  else:
    return layers.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
  """Strided 2-D convolution with 'SAME' padding.

  When stride > 1, then we do explicit zero-padding, followed by conv2d with
  'VALID' padding.

  Note that

     net = conv2d_same(inputs, num_outputs, 3, stride=stride)

  is equivalent to

     net = tf.contrib.layers.conv2d(inputs, num_outputs, 3, stride=1,
     padding='SAME')
     net = subsample(net, factor=stride)

  whereas

     net = tf.contrib.layers.conv2d(inputs, num_outputs, 3, stride=stride,
     padding='SAME')

  is different when the input's height or width is even, which is why we add the
  current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters.
    kernel_size: An int with the kernel_size of the filters.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    scope: Scope.

  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """
  if stride == 1:
    return layers_lib.conv2d(
        inputs,
        num_outputs,
        kernel_size,
        stride=1,
        rate=rate,
        padding='SAME',
        scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = array_ops.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return layers_lib.conv2d(
        inputs,
        num_outputs,
        kernel_size,
        stride=stride,
        rate=rate,
        padding='VALID',
        scope=scope)


@add_arg_scope
def stack_blocks_dense(net,
                       blocks,
                       output_stride=None,
                       outputs_collections=None):
  """Stacks ResNet `Blocks` and controls output feature density.

  First, this function creates scopes for the ResNet in the form of
  'block_name/unit_1', 'block_name/unit_2', etc.

  Second, this function allows the user to explicitly control the ResNet
  output_stride, which is the ratio of the input to output spatial resolution.
  This is useful for dense prediction tasks such as semantic segmentation or
  object detection.

  Most ResNets consist of 4 ResNet blocks and subsample the activations by a
  factor of 2 when transitioning between consecutive ResNet blocks. This results
  to a nominal ResNet output_stride equal to 8. If we set the output_stride to
  half the nominal network stride (e.g., output_stride=4), then we compute
  responses twice.

  Control of the output feature density is implemented by atrous convolution.

  Args:
    net: A `Tensor` of size [batch, height, width, channels].
    blocks: A list of length equal to the number of ResNet `Blocks`. Each
      element is a ResNet `Block` object describing the units in the `Block`.
    output_stride: If `None`, then the output will be computed at the nominal
      network stride. If output_stride is not `None`, it specifies the requested
      ratio of input to output spatial resolution, which needs to be equal to
      the product of unit strides from the start up to some level of the ResNet.
      For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,
      then valid values for the output_stride are 1, 2, 6, 24 or None (which
      is equivalent to output_stride=24).
    outputs_collections: Collection to add the ResNet block outputs.

  Returns:
    net: Output tensor with stride equal to the specified output_stride.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  # The current_stride variable keeps track of the effective stride of the
  # activations. This allows us to invoke atrous convolution whenever applying
  # the next residual unit would result in the activations having stride larger
  # than the target output_stride.
  current_stride = 1

  # The atrous convolution rate parameter.
  rate = 1

  for block in blocks:
    with variable_scope.variable_scope(block.scope, 'block', [net]) as sc:
      for i, unit in enumerate(block.args):
        if output_stride is not None and current_stride > output_stride:
          raise ValueError('The target output_stride cannot be reached.')

        with variable_scope.variable_scope('unit_%d' % (i + 1), values=[net]):
          # If we have reached the target output_stride, then we need to employ
          # atrous convolution with stride=1 and multiply the atrous rate by the
          # current unit's stride for use in subsequent layers.
          if output_stride is not None and current_stride == output_stride:
            net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
            rate *= unit.get('stride', 1)

          else:
            net = block.unit_fn(net, rate=1, **unit)
            current_stride *= unit.get('stride', 1)
      net = utils.collect_named_outputs(outputs_collections, sc.name, net)

  if output_stride is not None and current_stride != output_stride:
    raise ValueError('The target output_stride cannot be reached.')

  return net


def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  """Defines the default ResNet arg scope.

  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.

  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': ops.GraphKeys.UPDATE_OPS,
  }

  with arg_scope(
      [layers_lib.conv2d],
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      weights_initializer=initializers.variance_scaling_initializer(), 
      activation_fn=nn_ops.relu,
      normalizer_fn=layers.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([layers.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # tf.contrib.framework.arg_scope([tf.contrib.layers.max_pool2d], padding='VALID').
      with arg_scope([layers.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc


################################################End Fundamental functions#############################################

@slim.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None,
               use_bounded_activations=False):
  """Bottleneck residual unit variant with BN after convolutions.
  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
  its definition. Note that we use here the bottleneck variant which has an
  extra bottleneck layer.
  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.
    use_bounded_activations: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.
  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
      shortcut = subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(
          inputs,
          depth, [1, 1],
          stride=stride,
          activation_fn=tf.nn.relu6 if use_bounded_activations else None,
          scope='shortcut')

    residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           activation_fn=None, scope='conv3')

    if use_bounded_activations:
      # Use clip_by_value to simulate bandpass activation.
      residual = tf.clip_by_value(residual, -6.0, 6.0)
      output = tf.nn.relu6(shortcut + residual)
    else:
      output = tf.nn.relu(shortcut + residual)

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


def resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              store_non_strided_activations=False,
              reuse=None,
              scope=None):


  with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,stack_blocks_dense], 
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
          net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
        net = stack_blocks_dense(net, blocks, output_stride)

        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
          end_points['global_pool'] = net
        if num_classes:
          net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits')
          end_points[sc.name + '/logits'] = net
          if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            end_points[sc.name + '/spatial_squeeze'] = net
          end_points['predictions'] = slim.softmax(net, scope='predictions')
        return net, end_points
default_image_size = 224


def resnet_v1_block(scope, base_depth, num_units, stride):
  """Helper function for creating a resnet_v1 bottleneck block.
  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.
  Returns:
    A resnet_v1 bottleneck block.
  """
  return Block(scope, bottleneck, [{#  return resnet_utils.Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
}])

def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 store_non_strided_activations=False,
                 reuse=None,
                 scope='resnet_v1_50'):

  """ResNet-50 model"""
  blocks = [
      resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
      resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=store_non_strided_activations,
                   reuse=reuse, scope=scope)


with slim.arg_scope(resnet_arg_scope()):#with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    net, end_points = resnet_v1_50(inputs=input_layer,num_classes=12,is_training=False) 

print("net:",net,"end_points:", end_points)


outputf = slim.softmax(net)
#@@loss = tf.losses.mean_squared_error(label_oh, outputf, weights=1.0, scope=None)#, loss_collection=tf.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
#@loss = tf.reduce_mean(-tf.reduce_sum(label_oh * tf.log(outputf) + 1e-10, reduction_indices=[1]))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(label * tf.log(prediction), reduction_indices=[1]))
#sum_of_squares_loss = slim.losses.sum_of_squares(outp	ut, label_oh)
loss = slim.losses.softmax_cross_entropy(outputf, label_oh)
#@loss = tf.losses.softmax_cross_entropy(label_oh, output)
#loss = slim.losses.log_loss(output, label_oh, weights=1.0, epsilon=1e-7, scope=None):
#@trainer = tf.train.GradientDescentOptimizer(learning_rate=0.00000000001)
trainer = tf.train.AdamOptimizer(learning_rate=0.1)
update = trainer.minimize(loss)
print("outputf:",outputf, "outputf shape:", outputf.shape)
print("label_oh:",label_oh, "label_oh shape:", label_oh.shape) 
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
total_steps = 100001#50001#5001
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
    draw = range(14280)#range(408)#3270) #We load 10000 training examples in each training iteration.
#    pdb.set_trace()
    while i < total_steps:
         batch_index = np.random.choice(draw,size=batch_size,replace=False)
         x = X_train[:][batch_index]
#@         print("X_train:", X_train.shape, "X_train:",X_train )###        x = np.reshape(x,[batch_size,32,32,3],order='F')
         print("x :", x.shape, "x:",x )
         x = (x - np.mean(x,axis=0)) / np.std(x,axis=0)
         y = np.reshape(np.array(y_train[:])[batch_index],[batch_size,1])
         print("y:", y.shape, "y:",y)###        x = np.reshape(x,[batch_size,32,32,3],order='F')#@         prin###        pdb.set_trace()

#         if i < 50000:	
#              learning_rat = 0.1
#         elif i < 60000:
#              learning_rat = 0.01
#         elif i < 80000:
#              learning_rat = 0.001
#         else:#              learning_rat = 0.0001
         _,lossA,yP,LO = sess.run([update,loss,outputf,label_oh],feed_dict={input_layer:x,label_layer:np.hstack(y)})#We then run the train_step(update) operation, using feed_dict to replace the placeholder tensors x and y_ with the training examples. 
         print("yP:", yP.shape, "yP:",yP)
         print("LO:", LO.shape, "LO:",LO)
#         correct_prediction = tf.equal(np.argmax(y,1), np.argmax(yP,1))
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
         accuracy = np.sum(np.equal(np.argmax(LO,1), np.argmax(yP,1)))/float(len(y))
#         accuracy = np.sum(np.equal(np.hstack(y),np.argmax(yP,1)))/float(len(y))# Evaluate the Model: First we'll figure out where we predicted the correct label. np.hstack(y) is an extremely useful function which gives you the index of the highest entry in a tensor along some axis. For example, tf.argmax(y_,1) is the label our model thinks is most likely for each input, while np.hstack(y) is the true label. We can use tf.equal to check if our prediction matches the truth.
         print("outputf:",outputf,"outputf shape",outputf.shape,"label_oh",label_oh,"label_oh shape",label_oh.shape)
         print ("Train set accuracy: " + str(accuracy))
         l.append(lossA)
         a.append(accuracy)
         if i % 10 == 0:
              print("***************************** step % 10 ==0 **************************************************")
              print("Step: " + str(i) + " Loss: " + str(lossA) + " Accuracy: " + str(accuracy))
         if i % 100 == 0:
###            point = np.random.randint(0,10000-500)
            #point = np.random.randint(0,578-34)
              print("***************************** step % 100 ==0**************************************************")
              print ("Step: " + str(i)+ " Loss: " + str(lossA) + " Accuracy: " + str(accuracy))
              t1 = time.process_time()
              point = np.random.randint(0,2520-batch_size)           # point = np.random.randint(0,72-34)
###            xT = cifarT['data'][point:point+500]
              xT = X_test[:][point:point+batch_size]#34]
#@              print("XT before reshape:", xT.shape, "xT:",xT )
#            xT = np.reshape(xT,[500,32,32,3],order='F')
#@            xT = xT/np.float32(255.0)
#@            xT = (xT - np.mean(xT,axis=0)) / np.std(xT,axis=0)
#@            xT = (xT - np.mean(xT,axis=3))
#@              xT = np.reshape(xT,[batch_size,224,224,3],order='F') #            xT = np.reshape(xT,[34,224,224,3],order='F')
#@              print("XT after reshape:", xT.shape, "xT:",xT )
              print("mean for test data", np.mean(xT,axis=0 ) )
              print("std for test data", np.std(xT,axis=0) )
              xT = (xT - np.mean(xT,axis=0)) / np.std(xT,axis=0)
              print("xT after mean subtraction/std",xT.shape,"  ",xT[45,:,:,2] )
#@            xT = xT/np.float32(255.0)
#@            print("x after /255", xT.shape,"  ",xT[45,:,:,2] )
#@            xT = np.reshape(xT,[batch_size,224,224,3],order='F') #            xT = np.reshape(xT,[34,224,224,3],order='F')
		###            yT = np.reshape(np.array(cifarT['labels'])[point:point+500],[500])
              yT = np.reshape(np.array(y_test[:])[point:point+batch_size],[batch_size]) #        yT = np.reshape(np.array(y_test[:])[point:point+34],[34])
              print("Existing MOAs in batch testing data set (yT):" + str(yT))
              print( "Number of MOAs in batch testing data set:",collections.Counter(yT))
              lossT,yP_t,LOt = sess.run([loss,outputf,label_oh],feed_dict={input_layer:xT,label_layer:yT})
              print("yP_t:", yP_t.shape, "yP_t:",yP_t)
              print("LOt:", LOt.shape, "LOt:", LOt)
#              correct_predictiont = tf.equal(tf.argmax(yT,1), tf.argmax(yP_t,1))
#              accuracyt = tf.reduce_mean(tf.cast(correct_predictiont, "float"))
              accuracyt = np.sum(np.equal(np.argmax(LOt,1), np.argmax(yP_t,1)))/float(len(yT))
#@              accuracy = np.sum(np.equal(yT,np.argmax(yP,1)))/float(len(yT)) #We then invert the encoding by using the NumPy argmax() function on the first value in the sequence that returns the expected value 1 for the first integer.
              t = time.process_time()
              tnow.append(t)
              aT.append(accuracyt)
              print ("Test set accuracy: " + str(accuracyt))
              elapsed_time = time.process_time() - t1
              telaps.append(elapsed_time)      #elapse time for testing 60 samples 
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
              plt.savefig('/home/fmkazemi/code/result/time_vgg19_224' + str(i) + '.png') 
              print("Length Running time for test dataset", len(telaps))
              print("Running time for test dataset:" + str(telaps))
              print("Running time for each sample in test dataset:"+ str(elapsed_time/batch_size))#34))

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
              plt.savefig('/home/fmkazemi/code/result/entropyloss_vgg19_224' + str(i) + '.png') 
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
              plt.savefig('/home/fmkazemi/code/result/test_train_accuracy_vgg19_224' + str(i) + '.png')
	    ###############################plt.savefig('/home/farhad/Desktop/Mythesis/Backup_mycodes_Guillimin/code/result/test_accuracy.png')
	    ##plt.close()    # close the figure
              np.max(aT) #Best test accuracy
              ii=ii+1
	    #pdb.set_trace()
	    #nn_confMatrix = \
	    #    confusion_matrix(np.fromiter(chain.from_iterable(np.argmax(yP,1)),dtype=int),np.fromiter(chain.from_iterable(np.hstack	(y)),dtype=int))
	    #nn_confMatrix = confusion_matrix(np.argmax(yP,1),np.hstack(y))
              print("Prediction shape:")
              print(yP_t.shape)
              print("Predictions:")
              print(str(np.argmax(yP_t,1))) 
              print("True Label shape:",yT.shape)
              print("True Labels:")
              print(str(yT))
              del nn_confMatrix
              del nn_acc
            #print(np.argmax(yP,1))
              nn_confMatrix = confusion_matrix(np.argmax(yP_t,1),np.argmax(LOt,1))   #nn_confMatrix = confusion_matrix(np.argmax(yP,1),yT)  
              print("Confusion Matrix in step " + str(i))
              print(nn_confMatrix)
              nn_acc = np.sum(nn_confMatrix.diagonal()) / np.sum(nn_confMatrix,dtype=np.float)
              print("Accuracy based on Confusion Matrix: ",nn_acc)
              np.savetxt('/home/fmkazemi/code/result/ac.txt', a, fmt='%f')
              np.savetxt('/home/fmkazemi/code/result/act.txt', aT, fmt='%f')
              np.savetxt('/home/fmkazemi/code/result/loss.txt', l, fmt='%f')
              np.savetxt('/home/fmkazemi/code/result/time.txt', tnow, fmt='%d')	    	
         i+= 1	

