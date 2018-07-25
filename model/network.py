# Chris Abbott
#
# TensorFlow implementation of TinySSD
# https://arxiv.org/pdf/1802.06488.pdf

import tensorflow as tf

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

@add_arg_scope
def fire_module(inputs,
                squeeze_depth,
                expand1_depth,
                expand3_depth,
                name,
                reuse=None,
                scope=None):
  with tf.variable_scope(scope, name, [inputs], reuse=reuse):
    with arg_scope([conv2d, max_pool2d]):
      net = _squeeze(inputs, squeeze_depth)
      net = _expand(net, expand1_depth, expand3_depth)
    return net

def _squeeze(inputs, num_outputs):
  return conv2d(inputs, num_outputs, [1,1], stride=1, scope='squeeze')

def _expand(inputs, expand1_depth, expand3_depth):
  with tf.variable_scope('expand'):
    e1x1 = conv2d(inputs, expand1_depth, [1, 1], stride=1, scope='1x1')
    e3x3 = conv2d(inputs, expand3_depth, [3, 3], scope='3x3')
  return tf.concat([e1x1, e3x3], 1)

def tinySSD(images):

  # II. OPTIMIZED FIRE SUB-NETWORK STACK (first stack)

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,3,57],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(images, kernel, strides=[1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, 
                         ksize=[1,3,3,1], 
                         strides=[1,2,2,1],
                         padding='SAME',
                         name='pool1')

  # fire1
  fire1 = fire_module(conv1,
                      squeeze_depth=15,
                      expand1_depth=49,
                      expand3_depth=53,
                      name='fire1')

  # fire2
  fire2 = fire_module(fire1,
                      squeeze_depth=15,
                      expand1_depth=54,
                      expand3_depth=52,
                      name='fire2')

  # pool3
  pool3 = tf.nn.max_pool(fire2, 
                         ksize=[1,3,3,1], 
                         strides=[1,2,2,1],
                         padding='SAME',
                         name='pool3')

  # fire3
  fire3 = fire_module(pool3,
                      squeeze_depth=29,
                      expand1_depth=92,
                      expand3_depth=94,
                      name='fire3')

  # fire4
  fire4 = fire_module(fire3,
                      squeeze_depth=29,
                      expand1_depth=90,
                      expand3_depth=83,
                      name='fire4')

  # pool5
  pool5 = tf.nn.max_pool(fire4, 
                         ksize=[1,3,3,1], 
                         strides=[1,2,2,1],
                         padding='SAME',
                         name='pool5')


  # fire5
  fire5 = fire_module(pool5,
                      squeeze_depth=44,
                      expand1_depth=166,
                      expand3_depth=161,
                      name='fire5')

  # fire6
  fire6 = fire_module(fire5,
                      squeeze_depth=45,
                      expand1_depth=155,
                      expand3_depth=146,
                      name='fire6')

  # fire7
  fire7 = fire_module(fire6,
                      squeeze_depth=49,
                      expand1_depth=163,
                      expand3_depth=171,
                      name='fire7')

  # fire8
  fire8 = fire_module(fire7,
                      squeeze_depth=25,
                      expand1_depth=29,
                      expand3_depth=54,
                      name='fire8')

  # pool9
  pool9 = tf.nn.max_pool(fire8, 
                         ksize=[1,3,3,1], 
                         strides=[1,2,2,1],
                         padding='SAME',
                         name='pool9')

  # fire9
  fire9 = fire_module(pool9,
                      squeeze_depth=37,
                      expand1_depth=45,
                      expand3_depth=56,
                      name='fire9')

  # pool10
  pool10 = tf.nn.max_pool(fire9, 
                         ksize=[1,3,3,1], 
                         strides=[1,2,2,1],
                         padding='SAME',
                         name='pool10')

  # fire10
  fire10 = fire_module(pool10,
                       squeeze_depth=38,
                       expand1_depth=41,
                       expand3_depth=44,
                       name='fire10')

  # III. OPTIMIZED SUB-NETWORK STACK OF CONVOLUTIONAL FEATURE LAYERS (second stack)

  # conv12-1
  with tf.variable_scope('conv12_1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,4,51],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(fire10, kernel, strides=[1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv12_1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv12_1)

  # conv12-2
  with tf.variable_scope('conv12_2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,4,46],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(conv12_1, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv12_2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv12_2)

  # conv13-1
  with tf.variable_scope('conv13_1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,2,55],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(conv12_2, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv13_1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv13_1)

  # conv13-2
  with tf.variable_scope('conv13_2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,2,46],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(conv13_1, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv13_2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv13_2)


  # START OUTPUT PARAMETERS

  # fire5_mbox_loc
  with tf.variable_scope('fire5_mbox_loc') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,37,16],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(fire4, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    fire5_mbox_loc = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(fire5_mbox_loc)

  # fire5_mbox_conf  
  with tf.variable_scope('fire5_mbox_conf') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,37,84],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(fire4, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    fire5_mbox_conf = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(fire5_mbox_conf)

  # fire9_mbox_loc
  with tf.variable_scope('fire9_mbox_loc') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,18,24],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(fire8, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    fire9_mbox_loc = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(fire9_mbox_loc)

  # fire9_mbox_conf
  with tf.variable_scope('fire9_mbox_conf') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,18,126],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(fire8, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    fire9_mbox_conf = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(fire9_mbox_conf)

  # fire10_mbox_loc
  with tf.variable_scope('fire10_mbox_loc') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,9,24],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(fire9, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    fire10_mbox_loc = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(fire10_mbox_loc)

  # fire10_mbox_conf
  with tf.variable_scope('fire10_mbox_conf') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,9,126],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(fire9, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    fire10_mbox_conf = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(fire10_mbox_conf)

  # fire11_mbox_loc
  with tf.variable_scope('fire11_mbox_loc') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,4,24],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(fire10, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    fire11_mbox_loc = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(fire11_mbox_loc)

  # fire11_mbox_conf
  with tf.variable_scope('fire11_mbox_conf') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,4,126],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(fire10, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    fire11_mbox_conf = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(fire11_mbox_conf)

  # conv12_2_mbox_loc
  with tf.variable_scope('conv12_2_mbox_loc') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,2,24],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(conv12_2, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv12_2_mbox_loc = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv12_2_mbox_loc)

  # conv12_2_mbox_conf
  with tf.variable_scope('conv12_2_mbox_conf') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,2,126],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(conv12_2, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv12_2_mbox_conf = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv12_2_mbox_conf)

  # conv13_2_mbox_loc
  with tf.variable_scope('conv13_2_mbox_loc') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,1,16],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(conv13_2, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv13_2_mbox_loc = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv13_2_mbox_loc)

  # conv13_2_mbox_conf
  with tf.variable_scope('conv13_2_mbox_conf') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3,3,1,84],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(conv13_2, kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv13_2_mbox_conf = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv13_2_mbox_conf)

