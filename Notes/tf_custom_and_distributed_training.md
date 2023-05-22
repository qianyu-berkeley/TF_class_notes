# Custom and Distributed Training with TensorFlow

## 1. Understand the unit of Tensor

* Tensors are multi-dimensional arrays with a uniform data type. 
* Tensors are immutable.

Example:

```python
x = np.arange(0, 25)
x = tf.constant(x)
x = tf.square(x)
x = tf.reshape(x, (5, 5))
x = tf.cast(x, tf.float32)
```

* tf.constant vs tf.variable

    When using `tf.constant()`, you can pass in a 1D array (a vector) and set the `shape` parameter to turn this vector into a multi-dimensional array.  Whereas, with `tf.Variable()`, the shape of the tensor is derived from the shape given by the input array.  Setting `shape` to something other than `None` will not reshape a 1D array into a multi-dimensional array, and it will receive a `ValueError`.


*** 

## 2. Eager mode (vs Graph mode), Broadcasting, Operator overload, Numpy compatibility

Eager execution supports:

* Evaluate values immediately
* Support broadcasting
  * Broadcasting is an important mechanism to enable operate tensors with different shapes
* Support operator overloading
* Support Numpy compatibility

Example:

```python
# Immediate evaluation of a tensor whereas graph mode delays the calculation for performance improvement
x = 2
x_squared = tf.square(x)
print(f"results: {x_sqaured}")
>> result: 4

# Broadcasting
x = tf.reshape(x, (5, 5))
y = tf.constant(2, dtype=tf.float32)
result = tf.multiply(x, y)
result.shape # result is a 5x5 tensor

# Overloading python operator (i.e. we can use math operation for tensors)
a = tf.constant([[1, 2], [3, 4]])
a ** 2
>> [[1, 4][9, 16]], shape=(2, 2), dtype=int32)

# Numpy compatible
a = tf.constant(5)
b = tf.constant(6)
np.mulitply(a, b)
>> 15

# convert to numpy and back
ndarray = np.ones([3, 3])
tensor = tf.multiply(ndarray, 3)
tensor.numpy()
```
***

## 5. Graph Mode

* Graph mode enables tensorflow performance improvement
* `@tf.function` decorator is used to automatically generate graph-style code
* To view auto generated code use `tf.autograph.to_code(myfunction.python_functioin)`
* Tracing also behaves differently in graph mode, us `tf.print` which is a graph aware print
* Avoid defining variables (`tf.Variable`) inside the function for graph mode (`@tf.function`)

Examples:

```python
# simple function that returns the square if the input is greater than zero
@tf.function
def f(x):
    if x>0:
        x = x * x
    return x

print(tf.autograph.to_code(f.python_function))

# Fizzbuzz in autograph
@tf.function
def fizzbuzz(max_num):
    counter = 0
    for num in range(max_num):
        if num % 3 == 0 and num % 5 == 0:
            print('FizzBuzz')
        elif num % 3 == 0:
            print('Fizz')
        elif num % 5 == 0:
            print('Buzz')
        else:
            print(num)
        counter += 1
    return counter

print(tf.autograph.to_code(fizzbuzz.python_function))

# function with loop
@tf.function
def sum_even(items):
    s = 0
    for c in items:
        if c % 2 > 0:
            continue
        s += c
    return s

print(tf.autograph.to_code(sum_even.python_function))

@tf.function
def f(x):
    print("Traced with", x)
    # added tf.print
    tf.print("Executed with", x)

for i in range(5):
    f(2)
    
f(3)

# Need to define the variables outside of the decorated function
v = tf.Variable(1.0)

@tf.function
def f(x):
    return v.assign_add(x)

print(f(5))
```

*** 

## 3. Gradient Tape

* To perform forward and backward propagation
* `tf.GradientTape()` performs automatic differentiation
* `persistent=False` is the default behavior of `GradientTape` since we want it to be updated every epochs
    * When set to False (non-persistent), it can only be calculated once then it will expire.
* Higher order derivative can be performed by nesting `GradientTapes`
    * User needs to pay attention to proper indentation for the 2nd gradient calculation
* `watch()` function traces the tensor
* `gradient()` computes the gradient using operations recorded in context of this tap

Example: a simple gradient operations

```python
x = tf.constant(3.0)

# Notice that persistent is False by default
with tf.GradientTape() as t:
    t.watch(x)
    y = x * x
    z = y * y

# Compute dz/dx. 4 * x^3 at x = 3 --> 108.0
dz_dx = t.gradient(z, x)

# Set persistent=True so that you can reuse the tape
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = x * x
    z = y * y

# Compute dz/dx. 4 * x^3 at x = 3 --> 108.0
dz_dx = t.gradient(z, x)

# Run it again, it will work since it is persistent
dy_dx = t.gradient(y, x) 
```

Example: Higher order gradient operations

```python
x = tf.Variable(1.0)

with tf.GradientTape() as tape_2:
    with tf.GradientTape() as tape_1:
        y = x * x * x
    
    # The first gradient calculation should occur at least within the outer with block
    dy_dx = tape_1.gradient(y, x)
d2y_dx2 = tape_2.gradient(dy_dx, x)

# Alternative way of perform nested GradientTape
x = tf.Variable(1.0)

with tf.GradientTape() as tape_2:
    with tf.GradientTape() as tape_1:
        y = x * x * x
    
        # The first gradient calculation can also be within the inner with block
        dy_dx = tape_1.gradient(y, x)
d2y_dx2 = tape_2.gradient(dy_dx, x)

# The following indentation won't work
x = tf.Variable(1.0)

# Setting persistent=True still won't work
with tf.GradientTape(persistent=True) as tape_2:
    # Setting persistent=True still won't work
    with tf.GradientTape(persistent=True) as tape_1:
        y = x * x * x

# The first gradient call is outside the outer with block
# so the tape will expire after this
dy_dx = tape_1.gradient(y, x)

# the output will be `None`
d2y_dx2 = tape_2.gradient(dy_dx, x)

print(dy_dx)
print(d2y_dx2)
```

***

## 4. Custom Training Loop with Gradient Taping

Custom training Steps:

1. Setup data
2. Define a model
3. Define a optimizer and loss function
4. Define custom training loop functions
   1. Within `tf.GradientTape` context
   2. Calculate Loss
   3. Calculate derivative using `tape.gradient()` function (to back propergate)
   4. update trainable weights (pytorch would require to zero gradient but it is not needed in TF)
   5. apply reset_states() function for metric
5. Perform model custom training loop by applying the custom training function from step 4 for each epoch
6. Model evaluation

Example: MNIST fashion classification

```python
#------------------
# 1. Setup dataset
#------------------
# load dataset with tfds
train_data, info = tfds.load("fashion_mnist", split = "train", with_info = True, data_dir='./data/', download=False)
test_data = tfds.load("fashion_mnist", split = "test", data_dir='./data/', download=False)
class_names = ["T-shirt/top", "Trouser/pants", "Pullover shirt", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Alterntively create data set from tensor slice if we already have data in tensor
train_data = tf.data.Dataset.from_tensor_slices((norm_train_X.values, train_Y.values))
test_data = tf.data.Dataset.from_tensor_slices((norm_test_X.values, test_Y.values))


# Normalize image data
def format_image(data):        
    """prepare image data"""
    image = data["image"]
    image = tf.reshape(image, [-1]) # Flatten from (28, 28) to (748,)
    image = tf.cast(image, 'float32')
    image = image / 255.0
    return image, data["label"]

train_data = train_data.map(format_image)
test_data = test_data.map(format_image)

# create batches for train and test, note that only training needs to shuffle for randomness
batch_size = 64
train = train_data.shuffle(buffer_size=1024).batch(batch_size)
test =  test_data.batch(batch_size=batch_size)

#--------------------
# 2. Define a model
#--------------------
# Define model either using functional API or inherit model class
# This example use functional API
# Use an basic NN model for this image task
def base_model(): 
    """Create base model using functional API"""
    inputs = tf.keras.Input(shape=(784,), name='digits') 
    x = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(inputs) 
    x = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(x) 
    outputs = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(x) 
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

#------------------------------------------------
# 3. Define optimizer, loss function, and metric
#------------------------------------------------
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# multiclass classification with index target so use SparseCategoricalAccuracy
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

#-------------------------------------------
# 4. Define a custom training loop function
#-------------------------------------------
def apply_gradient(optimizer, model, x, y): 
    """Perform gradient calcuation (derivative), update weights"""
    with tf.GradientTape() as tape: 
        # we call it logits by convension but it is actually a probability distribution 
        # since we used softmax activation in the last layer of the model
        logits = model(x) 
        loss_value = loss_object(y_true=y, y_pred=logits)
  
    gradients = tape.gradient(loss_value, model.trainable_weights)

    # apply gradient to optimizer function
    optimizer.apply_gradients(zip(gradients, model.trainable_weights)) 
    return logits, loss_value

def train_data_for_one_epoch(): 
    """App gradient to training batches (one eopch), store losses of each batch in a list"""
    losses = []
    pbar = tqdm(total=len(list(enumerate(train))), position=0, leave=True, 
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')
    for step, (x_batch_train, y_batch_train) in enumerate(train):
        logits, loss_value = apply_gradient(optimizer, model, x_batch_train, y_batch_train)
        losses.append(loss_value)
        train_acc_metric(y_batch_train, logits)
        pbar.set_description("Training loss for step %s: %.4f" % (int(step), float(loss_value)))
        pbar.update()
    return losses

def perform_validation():
    """Perform validation, save losses of each batch to a list"""
    losses = []
    for x_val, y_val in test:
        val_logits = model(x_val)
        val_loss = loss_object(y_true=y_val, y_pred=val_logits)
        losses.append(val_loss)
        val_acc_metric(y_val, val_logits)
    return losses

#---------------------------------------
# 5. Perform Model Training with epochs
# Within each epochs
#---------------------------------------
# measure the avg training loss and validation loss and report
# reset metric at the end of each epochs
model = base_model()

epochs = 10
epochs_val_losses, epochs_train_losses = [], []

for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))
    
    losses_train = train_data_for_one_epoch()
    train_acc = train_acc_metric.result()

    losses_val = perform_validation()
    val_acc = val_acc_metric.result()

    losses_train_mean = np.mean(losses_train)
    losses_val_mean = np.mean(losses_val)
    epochs_val_losses.append(losses_val_mean)
    epochs_train_losses.append(losses_train_mean)

    print('\n Epoch %s: Train loss: %.4f  Validation Loss: %.4f, Train Accuracy: %.4f, Validation Accuracy %.4f' % (epoch, float(losses_train_mean), float(losses_val_mean), float(train_acc), float(val_acc)))
    
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()

#-----------------------------------------
# 6. Model evaluation or make predictions
#-----------------------------------------
def plot_metrics(train_metric, val_metric, metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0,ylim)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.plot(train_metric,color='blue',label=metric_name)
    plt.plot(val_metric,color='green',label='val_' + metric_name)

plot_metrics(epochs_train_losses, epochs_val_losses, "Loss", "Loss", ylim=1.0)

test_outputs = model(norm_test_X.values)
plot_confusion_matrix(test_Y.values, tf.round(test_outputs), title='Confusion Matrix for Untrained Model')
```

#### Note:

`CategoricalCrossentropy` vs `SparseCategoricalCrossentroy`

* Both computes crossentropy loss between labels and predictions
* Both work with two or more label classes
* `CategoricalCrossentropy` expects label to `one-hot` encoded whereas `SparseCategoricalCrossentropy` expect label to integer
* `from_logits` flag determine whether `y_pred` is expected to be a logits tensor or not, the default is `False` meaning y_pred is encodeds a probability distribution.
  * `from_logits`=True attribute inform the loss function that the output values generated by the model are not normalized, a.k.a. logits. In other words, the softmax function has not been applied on them to produce a probability distribution.


***

## Custom metrics
We can create custom metric function by Inherit `tf.keras.metrics.Metric` class
* We need to define a `reset_states()` class function to reset metric state at beginning of each epochs

Example

```python
class F1Score(tf.keras.metrics.Metric):

    def __init__(self, name='f1_score', **kwargs):
        '''initializes attributes of the class'''
        
        # call the parent class init
        super(F1Score, self).__init__(name=name, **kwargs)

        # Initialize Required variables
        # true positives
        self.tp = tf.Variable(0, dtype = 'int32')
        # false positives
        self.fp = tf.Variable(0, dtype = 'int32')
        # true negatives
        self.tn = tf.Variable(0, dtype = 'int32')
        # false negatives
        self.fn = tf.Variable(0, dtype = 'int32')

    def update_state(self, y_true, y_pred, sample_weight=None):
        '''
        Accumulates statistics for the metric
        
        Args:
            y_true: target values from the test data
            y_pred: predicted values by the model
        '''

        # Calulcate confusion matrix.
        conf_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=2)
        
        # Update values of true positives, true negatives, false positives and false negatives from confusion matrix.
        self.tn.assign_add(conf_matrix[0][0])
        self.tp.assign_add(conf_matrix[1][1])
        self.fp.assign_add(conf_matrix[0][1])
        self.fn.assign_add(conf_matrix[1][0])

    def result(self):
        '''Computes and returns the metric value tensor.'''

        # Calculate precision
        if (self.tp + self.fp == 0):
            precision = 1.0
        else:
            precision = self.tp / (self.tp + self.fp)
      
        # Calculate recall
        if (self.tp + self.fn == 0):
            recall = 1.0
        else:
            recall = self.tp / (self.tp + self.fn)

        f1_score = 2*((precision * recall)/(precision + recall)) 
        return f1_score

    def reset_states(self):
        '''Resets all of the metric state variables.'''
        
        # The state of the metric will be reset at the start of each epoch.
        self.tp.assign(0)
        self.tn.assign(0) 
        self.fp.assign(0)
        self.fn.assign(0)
```

***

## Tensorflow Distributed Strategies (GPU)

**Use the following steps if one want to use build-in functions such as `fit`, `compile`**:

1. Define a distribution strategy based on hardware and check hardware configuration matches
2. create data batch based on the strategy (num of replicas) or `experimental_distributed_dataset`
3. Define model under `strategy.scope()` context
4. compile and fit

**Use the following steps if one want to use a custom training loop**

1. Define a desired distribution strategy based on hardware and check hardware configuration matches
2. Prepare training and validation data
   1. Setup up batches
   2. based on the strategy, create replica or set data to be distributed `strategy.experimental_distribute_dataset()`
3. Under `strategy.scope()`:
   1. Define model
   2. Define loss function
   3. Define opimizer
   4. Essentially a whole training/validation step (Add `@tf.fuction` decorator to create distributed train/validate steps)
4. Create training loop using the functions defined in 3

Refer to [The strategy.scope() Reference](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy#scope) for details

**Available Strateges**

* MirroredStrategy
* TPUStrategy
* MultiWorkerMirroredStrategy
* CentralStorageStrategy

### MirroredStrategy with build-in functions (`fit`, `compile`)

* Ideal for a single instance with mutliple GPUs, the replicated of variables are send to GPUs
* Use `BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync` to calculate batch_size
* Wrap model definition with `strategy.scope()` context
* Apply build-in function such as `fit`

Example: A simple case

```python
# Define the strategy to use and print the number of devices found
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Get the number of examples in the train and test sets
num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000

BATCH_SIZE_PER_REPLICA = 64
# Use for Mirrored Strategy
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# Function for normalizing the image
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

# Set up the train and eval data set
train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

# Use for Mirrored Strategy -- comment out `with strategy.scope():` and deindent for no strategy
with strategy.scope():
    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
    ])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

model.fit(train_dataset, epochs=12)
```

### MirroredStrategy with custom training loop

* Define `cross_device_ops` in MirroedStrategy 
* Create train/test dataset with `strategy.experimental_distribute_dataset()`
* set `reduction=tf.keras.losses.Reduction.NONE` in the loss function so we do the reduction afterwards and divide by global batch size.
* Define model object and loss function under with `strategy.scope()` context


Example: `MirroredStrategy` with multi-GPU but different GPU types 

```python
# Define the minimum number cores for GPU
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"

# `tf.distribute.MirroredStrategy` will auto-detected device.
# If you have different types of GPUs in your system, you have to set up cross_device_ops
strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Get the data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Adding a dimension to the array -> new shape == (28, 28, 1)
# We are doing this because the first layer in our model is a convolutional
# layer and it requires a 4D input (batch_size, height, width, channels).
train_images = train_images[..., None]
test_images = test_images[..., None]

# Normalize the images to [0, 1] range.
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

# Batch the input data
BUFFER_SIZE = len(train_images)
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# Create Datasets from the batches
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

# Create Distributed Datasets from the datasets
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


# Create the model architecture
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
        ])
    return model

with strategy.scope():
    # We will use sparse categorical crossentropy as always. But, instead of having the loss function
    # manage the map reduce across GPUs for us, we'll do it ourselves.
    # Set reduction to `none` so we can do the reduction afterwards and divide byglobal batch size.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def compute_loss(labels, predictions):
        # Compute Loss uses the loss object to compute the loss
        # Notice that per_example_loss will have an entry per GPU
        # so in this case there'll be 2 -- i.e. the loss for each replica
        per_example_loss = loss_object(labels, predictions)
        # You can print it to see it -- you'll get output like this:
        # Tensor("sparse_categorical_crossentropy/weighted_loss/Mul:0", shape=(48,), dtype=float32, device=/job:localhost/replica:0/task:0/device:GPU:0)
        # Tensor("replica_1/sparse_categorical_crossentropy/weighted_loss/Mul:0", shape=(48,), dtype=float32, device=/job:localhost/replica:0/task:0/device:GPU:1)
        # Note in particular that replica_0 isn't named in the weighted_loss -- the first is unnamed, the second is replica_1 etc
        print(per_example_loss)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

    # We'll just reduce by getting the average of the losses
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    # Accuracy on train and test will be SparseCategoricalAccuracy
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # Optimizer will be Adam
    optimizer = tf.keras.optimizers.Adam()

    # Create the model within the scope
    model = create_model()

# `run` replicates the provided computation and runs it
# with the distributed input.
def train_step(inputs):
    images, labels = inputs
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)
    return loss

@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    #tf.print(per_replica_losses.values)
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


def test_step(inputs):
    images, labels = inputs

    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss.update_state(t_loss)
    test_accuracy.update_state(labels, predictions)

@tf.function
def distributed_test_step(dataset_inputs):
    return strategy.run(test_step, args=(dataset_inputs,))


# Training Loop
EPOCHS = 10
for epoch in range(EPOCHS):
    # Do Training
    total_loss = 0.0
    num_batches = 0
    for batch in train_dist_dataset:
        total_loss += distributed_train_step(batch)
        num_batches += 1
    train_loss = total_loss / num_batches

    # Do Testing
    for batch in test_dist_dataset:
        distributed_test_step(batch)

    template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, " "Test Accuracy: {}")

    print (template.format(epoch+1, train_loss, train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))

    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
```

###  TPUStrategy

* Only available for colab or google service, need to detect the hardware
* The example below also demo the tfrecord format

Example:
```python
# Detect TPU hardware
try:
    tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_address) # TPU detection
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu) 
    # Going back and forth between TPU and host is expensive.
    # Better to run 128 batches on the TPU before reporting back.
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])  
    print("Number of accelerators: ", strategy.num_replicas_in_sync)
except ValueError:
    print('TPU failed to initialize.')

SIZE = 224 #@param ["192", "224", "331", "512"] {type:"raw"}
IMAGE_SIZE = [SIZE, SIZE]

# Dataset info
GCS_PATTERN = 'gs://flowers-public/tfrecords-jpeg-{}x{}/*.tfrec'.format(IMAGE_SIZE[0], IMAGE_SIZE[1])

BATCH_SIZE = 128  # On TPU in Keras, this is the per-core batch size. The global batch size is 8x this.

VALIDATION_SPLIT = 0.2
# Class do not change, maps to the labels in the data (folder names)
CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'] 

# splitting data files between training and validation
filenames = tf.io.gfile.glob(GCS_PATTERN)
random.shuffle(filenames)

split = int(len(filenames) * VALIDATION_SPLIT)
training_filenames = filenames[split:]
validation_filenames = filenames[:split]
print("Pattern matches {} data files. Splitting dataset into {} training files and {} validation files".format(len(filenames), len(training_filenames), len(validation_filenames)))

validation_steps = int(3670 // len(filenames) * len(validation_filenames)) // BATCH_SIZE
steps_per_epoch = int(3670 // len(filenames) * len(training_filenames)) // BATCH_SIZE
print("With a batch size of {}, there will be {} batches per training epoch and {} batch(es) per validation run.".format(BATCH_SIZE, steps_per_epoch, validation_steps))

def read_tfrecord(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar
        "one_hot_class": tf.io.VarLenFeature(tf.float32),
    }
    example = tf.io.parse_single_example(example, features)
    image = example['image']
    class_label = example['class']
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    class_label = tf.cast(class_label, tf.int32)
    return image, class_label

def load_dataset(filenames):
    # read from TFRecords. For optimal performance, use "interleave(tf.data.TFRecordDataset, ...)"
    # to read from multiple TFRecord files at once and set the option experimental_deterministic = False
    # to allow order-altering optimizations.

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO) # faster
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    return dataset

def get_batched_dataset(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False) # drop_remainder will be needed on TPU
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_training_dataset():
    dataset = get_batched_dataset(training_filenames)
    dataset = strategy.experimental_distribute_dataset(dataset)
    return dataset

def get_validation_dataset():
    dataset = get_batched_dataset(validation_filenames)
    dataset = strategy.experimental_distribute_dataset(dataset)
    return dataset

# Define a model by inheriting the Model class
class MyModel(tf.keras.Model):
    def __init__(self, classes):
        super(MyModel, self).__init__()
        self._conv1a = tf.keras.layers.Conv2D(kernel_size=3, filters=16, padding='same', activation='relu')
        self._conv1b = tf.keras.layers.Conv2D(kernel_size=3, filters=30, padding='same', activation='relu')
        self._maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2)
        
        self._conv2a = tf.keras.layers.Conv2D(kernel_size=3, filters=60, padding='same', activation='relu')
        self._maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2)
        
        self._conv3a = tf.keras.layers.Conv2D(kernel_size=3, filters=90, padding='same', activation='relu')
        self._maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=2)
        
        self._conv4a = tf.keras.layers.Conv2D(kernel_size=3, filters=110, padding='same', activation='relu')
        self._maxpool4 = tf.keras.layers.MaxPooling2D(pool_size=2)
        
        self._conv5a = tf.keras.layers.Conv2D(kernel_size=3, filters=130, padding='same', activation='relu')
        self._conv5b = tf.keras.layers.Conv2D(kernel_size=3, filters=40, padding='same', activation='relu')
        
        self._pooling = tf.keras.layers.GlobalAveragePooling2D()
        self._classifier = tf.keras.layers.Dense(classes, activation='softmax')

    def call(self, inputs):
        x = self._conv1a(inputs)
        x = self._conv1b(x)
        x = self._maxpool1(x)

        x = self._conv2a(x)
        x = self._maxpool2(x)

        x = self._conv3a(x)
        x = self._maxpool3(x)

        x = self._conv4a(x)
        x = self._maxpool4(x)

        x = self._conv5a(x)
        x = self._conv5b(x)

        x = self._pooling(x)
        x = self._classifier(x)
        return x

with strategy.scope():
    model = MyModel(classes=len(CLASSES))
    # Set reduction to `none` so we can do the reduction afterwards and divide by
    # global batch size.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)

    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE * strategy.num_replicas_in_sync)

    test_loss = tf.keras.metrics.Mean(name='test_loss')

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')
    
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step,args=(dataset_inputs,))
        print(per_replica_losses)
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                            axis=None)
    
    @tf.function
    def distributed_test_step(dataset_inputs):
        strategy.run(test_step, args=(dataset_inputs,))


    def train_step(inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
        predictions = model(images)
        loss = compute_loss(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_accuracy.update_state(labels, predictions)

        return loss 

    def test_step(inputs):
        images, labels = inputs

        predictions = model(images)
        loss = loss_object(labels, predictions)

        test_loss.update_state(loss)
        test_accuracy.update_state(labels, predictions)

EPOCHS = 40
with strategy.scope():
    for epoch in range(EPOCHS):
        # TRAINING LOOP
        total_loss = 0.0
        num_batches = 0
        for x in get_training_dataset():
        total_loss += distributed_train_step(x)
        num_batches += 1
        train_loss = total_loss / num_batches

        # TESTING LOOP
        for x in get_validation_dataset():
        distributed_test_step(x)

        template = ("Epoch {}, Loss: {:.2f}, Accuracy: {:.2f}, Test Loss: {:.2f}, "
                    "Test Accuracy: {:.2f}")
        print (template.format(epoch+1, train_loss,
                            train_accuracy.result()*100, test_loss.result() / strategy.num_replicas_in_sync,
                            test_accuracy.result()*100))

        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()
```


## Reference:

* [Stanford CS231n CNN for visual recognition notes](https://cs231n.github.io/neural-networks-3/)
* [tf.keras.metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics)
* [Fizz Buzz Test](http://wiki.c2.com/?FizzBuzzTest)
* [Multi-worker training with Keras](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras)
* [strategy.scope()](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy#scope)