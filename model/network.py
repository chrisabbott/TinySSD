# Chris Abbott
#
# TensorFlow implementation of TinySSD
# https://arxiv.org/pdf/1802.06488.pdf

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('use_fp16', False, 
                            "Train the model using 16-bit floating points")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            "Batch size for training.")
tf.app.flags.DEFINE_integer('num_classes', 21,
                            "Number of classes in dataset.")

def _activation_summary(x):
  tf.summary.histogram("%s activations" % x.op.name, x)
  tf.summary.scalar("%s sparsity" % x.op.name, tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
    name,
    shape,
    tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_loss")
    tf.add_to_collection('losses', weight_decay)
  return var

def conv2d(inputs,
           filters,
           kernel_size,
           strides=(1, 1),
           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
           bias_initializer=tf.zeros_initializer(),
           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002),
           name=None):
  return tf.layers.conv2d(
      inputs,
      filters,
      kernel_size,
      strides,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      activation=tf.nn.relu,
      name=name,
      padding="same")

def fire_module(inputs, squeeze_depth, expand1_depth, expand3_depth, name):
  """Fire module: squeeze input filters, then apply spatial convolutions."""
  with tf.variable_scope(name, "fire", [inputs]):
    squeezed = conv2d(inputs, squeeze_depth, [1, 1], name="squeeze")
    e1x1 = conv2d(squeezed, expand1_depth, [1, 1], name="e1x1")
    e3x3 = conv2d(squeezed, expand3_depth, [3, 3], name="e3x3")
    return tf.concat([e1x1, e3x3], axis=3)

def _tensor_shape(x, rank=3):
  if x.get_shape().is_fully_defined():
    return x.get_shape().as_list()
  else:
    static_shape = x.get_shape().with_rank(rank).as_list()
    dynamic_shape = tf.unstack(tf.shape(x), rank)
    return [s if s is not None else d
            for s, d in zip(static_shape, dynamic_shape)]

def SSD_multibox_layer(inputs, 
                       num_classes, 
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
  outputs = inputs

  # Location predictions
  num_anchors = len(sizes) + len(ratios)
  num_loc_pred = num_anchors * 4

  with tf.variable_scope('loc_pred') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,INPUTDIMS,num_loc_pred],
                                         stddev=5e-2,
                                         wd=None)
    loc_pred = tf.nn.conv2d(outputs, kernel)
    # custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred, _tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])

  
  # Class predictions
  num_cls_pred = num_anchors * num_classes

  with tf.variable_scope('cls_pred') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,INPUTDIMS,num_cls_pred],
                                         stddev=5e-2,
                                         wd=None)
    cls_pred = tf.nn.conv2d(outputs, kernel)
    # custom_layers.channel_to_last(loc_pred)
    cls_pred = tf.reshape(cls_pred, _tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])

  return cls_pred, loc_pred

def tinySSD(images):

  # List of endpoints
  feature_layers = ['fire5_mbox_loc',
                    'fire5_mbox_conf',
                    'fire9_mbox_loc',
                    'fire9_mbox_conf',
                    'fire10_mbox_loc',
                    'fire10_mbox_conf',
                    'fire11_mbox_loc',
                    'fire11_mbox_conf',
                    'conv12_2_mbox_loc',
                    'conv12_2_mbox_conf',
                    'conv13_2_mbox_loc',
                    'conv13_2_mbox_conf']

  # II. OPTIMIZED FIRE SUB-NETWORK STACK (first stack)

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,3,57],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(images, kernel, strides=[1, 2, 2, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [57], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)
  print("conv1: %s" % conv1.get_shape())

  # pool1
  pool1 = tf.nn.max_pool(conv1, 
                         ksize=[1,3,3,1], 
                         strides=[1,2,2,1],
                         padding='VALID',
                         name='pool1')
  print("pool1: %s" % pool1.get_shape())

  # fire1
  fire1 = fire_module(pool1,
                      squeeze_depth=15,
                      expand1_depth=49,
                      expand3_depth=53,
                      name='fire1')
  print("fire1: %s" % fire1.get_shape())

  # fire2
  fire2 = fire_module(fire1,
                      squeeze_depth=15,
                      expand1_depth=54,
                      expand3_depth=52,
                      name='fire2')
  print("fire2: %s" % fire2.get_shape())

  # pool3
  pool3 = tf.nn.max_pool(fire2, 
                         ksize=[1,3,3,1], 
                         strides=[1,2,2,1],
                         padding='SAME',
                         name='pool3')
  print("pool3: %s" % pool3.get_shape())

  # fire3
  fire3 = fire_module(pool3,
                      squeeze_depth=29,
                      expand1_depth=92,
                      expand3_depth=94,
                      name='fire3')
  print("fire3: %s" % fire3.get_shape())

  # fire4
  fire4 = fire_module(fire3,
                      squeeze_depth=29,
                      expand1_depth=90,
                      expand3_depth=83,
                      name='fire4')
  print("fire4: %s" % fire4.get_shape())

  # pool5
  pool5 = tf.nn.max_pool(fire4, 
                         ksize=[1,3,3,1], 
                         strides=[1,2,2,1],
                         padding='VALID',
                         name='pool5')
  print("pool5: %s" % pool5.get_shape())

  # fire5
  fire5 = fire_module(pool5,
                      squeeze_depth=44,
                      expand1_depth=166,
                      expand3_depth=161,
                      name='fire5')
  print("fire5: %s" % fire5.get_shape())

  # fire6
  fire6 = fire_module(fire5,
                      squeeze_depth=45,
                      expand1_depth=155,
                      expand3_depth=146,
                      name='fire6')
  print("fire6: %s" % fire6.get_shape())

  # fire7
  fire7 = fire_module(fire6,
                      squeeze_depth=49,
                      expand1_depth=163,
                      expand3_depth=171,
                      name='fire7')
  print("fire7: %s" % fire7.get_shape())

  # fire8
  fire8 = fire_module(fire7,
                      squeeze_depth=25,
                      expand1_depth=29,
                      expand3_depth=54,
                      name='fire8')
  print("fire8: %s" % fire8.get_shape())

  # pool9
  pool9 = tf.nn.max_pool(fire8, 
                         ksize=[1,3,3,1], 
                         strides=[1,2,2,1],
                         padding='SAME',
                         name='pool9')
  print("pool9: %s" % pool9.get_shape())

  # fire9
  fire9 = fire_module(pool9,
                      squeeze_depth=37,
                      expand1_depth=45,
                      expand3_depth=56,
                      name='fire9')
  print("fire9: %s" % fire9.get_shape())

  # pool10
  pool10 = tf.nn.max_pool(fire9, 
                         ksize=[1,3,3,1], 
                         strides=[1,2,2,1],
                         padding='VALID',
                         name='pool10')
  print("pool10: %s" % pool10.get_shape())

  # fire10
  fire10 = fire_module(pool10,
                       squeeze_depth=38,
                       expand1_depth=41,
                       expand3_depth=44,
                       name='fire10')
  print("fire10: %s" % fire10.get_shape())

  # III. OPTIMIZED SUB-NETWORK STACK OF CONVOLUTIONAL FEATURE LAYERS (second stack)

  # conv12-1
  with tf.variable_scope('conv12_1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,85,51],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(fire10, kernel, strides=[1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [51], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv12_1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv12_1)
    print("conv12-1: %s" % conv12_1.get_shape())

  # conv12-2
  with tf.variable_scope('conv12_2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,51,46],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(conv12_1, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [46], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv12_2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv12_2)
    print("conv12-2: %s" % conv12_2.get_shape())

  # conv13-1
  with tf.variable_scope('conv13_1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,46,55],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(conv12_2, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [55], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv13_1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv13_1)
    print("conv13-1: %s" % conv13_1.get_shape())

  # conv13-2
  with tf.variable_scope('conv13_2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,55,46],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(conv13_1, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [46], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv13_2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv13_2)
    print("conv13-2: %s" % conv13_2.get_shape())


  # START OUTPUT PARAMETERS

  endPoints = {}

  # fire5_mbox_loc
  with tf.variable_scope('fire5_mbox_loc') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,173,16],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(fire4, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    fire5_mbox_loc = tf.nn.relu(pre_activation, name=scope.name)
    endPoints["fire5_mbox_loc"] = fire5_mbox_loc
    _activation_summary(fire5_mbox_loc)
    print("fire5_mbox_loc: %s" % fire5_mbox_loc.get_shape())

  # fire5_mbox_conf  
  with tf.variable_scope('fire5_mbox_conf') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,173,84],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(fire4, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [84], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    fire5_mbox_conf = tf.nn.relu(pre_activation, name=scope.name)
    endPoints["fire5_mbox_conf"] = fire5_mbox_conf
    _activation_summary(fire5_mbox_conf)
    print("fire5_mbox_conf: %s" % fire5_mbox_conf.get_shape())

  # fire9_mbox_loc
  with tf.variable_scope('fire9_mbox_loc') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,83,24],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(fire8, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [24], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    fire9_mbox_loc = tf.nn.relu(pre_activation, name=scope.name)
    endPoints["fire9_mbox_loc"] = fire9_mbox_loc
    _activation_summary(fire9_mbox_loc)
    print("fire9_mbox_loc: %s" % fire9_mbox_loc.get_shape())

  # fire9_mbox_conf
  with tf.variable_scope('fire9_mbox_conf') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,83,126],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(fire8, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [126], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    fire9_mbox_conf = tf.nn.relu(pre_activation, name=scope.name)
    endPoints["fire9_mbox_conf"] = fire9_mbox_conf
    _activation_summary(fire9_mbox_conf)
    print("fire9_mbox_conf: %s" % fire9_mbox_conf.get_shape())

  # fire10_mbox_loc
  with tf.variable_scope('fire10_mbox_loc') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,101,24],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(fire9, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [24], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    fire10_mbox_loc = tf.nn.relu(pre_activation, name=scope.name)
    endPoints["fire10_mbox_loc"] = fire10_mbox_loc
    _activation_summary(fire10_mbox_loc)
    print("fire10_mbox_loc: %s" % fire10_mbox_loc.get_shape())

  # fire10_mbox_conf
  with tf.variable_scope('fire10_mbox_conf') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,101,126],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(fire9, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [126], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    fire10_mbox_conf = tf.nn.relu(pre_activation, name=scope.name)
    endPoints["fire10_mbox_conf"] = fire10_mbox_conf
    _activation_summary(fire10_mbox_conf)
    print("fire10_mbox_conf: %s" % fire10_mbox_conf.get_shape())

  # fire11_mbox_loc
  with tf.variable_scope('fire11_mbox_loc') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,85,24],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(fire10, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [24], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    fire11_mbox_loc = tf.nn.relu(pre_activation, name=scope.name)
    endPoints["fire11_mbox_loc"] = fire11_mbox_loc
    _activation_summary(fire11_mbox_loc)
    print("fire11_mbox_loc: %s" % fire11_mbox_loc.get_shape())

  # fire11_mbox_conf
  with tf.variable_scope('fire11_mbox_conf') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,85,126],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(fire10, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [126], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    fire11_mbox_conf = tf.nn.relu(pre_activation, name=scope.name)
    endPoints["fire11_mbox_conf"] = fire11_mbox_conf
    _activation_summary(fire11_mbox_conf)
    print("fire11_mbox_conf: %s" % fire11_mbox_conf.get_shape())

  # conv12_2_mbox_loc
  with tf.variable_scope('conv12_2_mbox_loc') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,46,24],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(conv12_2, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [24], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv12_2_mbox_loc = tf.nn.relu(pre_activation, name=scope.name)
    endPoints["conv12_2_mbox_loc"] = conv12_2_mbox_loc
    _activation_summary(conv12_2_mbox_loc)
    print("conv12_2_mbox_loc: %s" % conv12_2_mbox_loc.get_shape())

  # conv12_2_mbox_conf
  with tf.variable_scope('conv12_2_mbox_conf') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,46,126],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(conv12_2, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [126], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv12_2_mbox_conf = tf.nn.relu(pre_activation, name=scope.name)
    endPoints["conv12_2_mbox_conf"] = conv12_2_mbox_conf
    _activation_summary(conv12_2_mbox_conf)
    print("conv12_2_mbox_conf: %s" % conv12_2_mbox_conf.get_shape())

  # conv13_2_mbox_loc
  with tf.variable_scope('conv13_2_mbox_loc') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,46,16],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(conv13_2, kernel, strides=[1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv13_2_mbox_loc = tf.nn.relu(pre_activation, name=scope.name)
    endPoints["conv13_2_mbox_loc"] = conv13_2_mbox_loc
    _activation_summary(conv13_2_mbox_loc)
    print("conv13_2_mbox_loc: %s" % conv13_2_mbox_loc.get_shape())

  # conv13_2_mbox_conf
  with tf.variable_scope('conv13_2_mbox_conf') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,46,84],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(conv13_2, kernel, strides=[1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [84], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv13_2_mbox_conf = tf.nn.relu(pre_activation, name=scope.name)
    endPoints["conv13_2_mbox_conf"] = conv13_2_mbox_conf
    _activation_summary(conv13_2_mbox_conf)
    print("conv13_2_mbox_conf: %s" % conv13_2_mbox_conf.get_shape())


