
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import Callback, EarlyStopping

from  model import PSPNet50
from preprocess import generate_arrays_from_file, generate_arrays_from_images, load_dataset

import os
import datetime

started_time = datetime.datetime.now()
started_time_format =  '{0:%m%d_%H%M_started}'.format(started_time)

#define crossentropy
def crossentropy(y_true, y_pred) :
    delta = 1e-7
    return -K.sum(y_true*K.log(y_pred + delta) )

#datasets path, result path, weight path
training_dataset_dir = "./Dataset/train/image/"
training_label_dir   = "./Dataset//train/label/"

validation_dataset_dir = "./Dataset/validation/image/"
validation_label_png_dir   = "./Dataset/validation/label/"

result_dir  = "./results/"
weights_dir  = "./weights/"
log_dir = "./tflog" + started_time_format + "/"

weights_name = weights_dir + "PSPNet50_params" + "_" + started_time_format + ".h5"

#make result path
if os.path.exists(result_dir) == False :
    os.mkdir(result_dir)

#make weights path
if os.path.exists(weights_dir) == False :
    os.mkdir(weights_dir)

#Input_shape
input_shape = (0, 0, 3)

#epoches
num_epoches = 0

#batchsize
num_batchsizes = 0

#number of train samples
num_of_train_samples = 0

#number of validation samples
num_of_validation_samples = 0

#number of steps per epoch
num_steps_per_epoch = int(num_of_train_samples / num_batchsizes)

#number of validation steps
num_val_steps = int(num_of_validation_samples / num_batchsizes )

#Number of categories
df = pd.read_csv("categories.tsv", delimiter="\t", index_col=0)
NUM_OF_CLASSES = len(df)

# ### add for TensorBoard
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

old_session = KTF.get_session()

session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)
# ### end

#data and labels
training_data_list = os.listdir(training_dataset_dir)
train_list = []
for data in training_data_list :
    train_list.append(data.split(".")[0])

#data and labels
validation_data_list = os.listdir(validation_dataset_dir)
validation_list = []
for data in validation_data_list :
    validation_list.append(data.split(".")[0])

model = PSPNet50(input_shape=input_shape, NUM_OF_CLASSES=NUM_OF_CLASSES)


model.compile(loss=crossentropy, optimizer='adagrad', metrics=["accuracy"])

# save model's architecture
model_json_str = model.to_json()


### add for TensorBoard
tb_cb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
### end

#figure model's summary
#model.summary()

#plot model's overview
plot_model(model, show_shapes=True, to_file='model.png')

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

cbks = [tb_cb, early_stopping]
#cbks = [tb_cb]

#training process
model.fit_generator(generator=load_dataset(train_list, training_dataset_dir, training_label_png_dir,  (input_shape[0], input_shape[1]), NUM_OF_CLASSES, num_batchsizes),
       steps_per_epoch = num_steps_per_epoch,
       nb_epoch=num_epoches,
       validation_data=load_dataset(validation_list, validation_dataset_dir, validation_label_png_dir,  (input_shape[0], input_shape[1]), NUM_OF_CLASSES, num_batchsizes),
       validation_steps=num_val_steps,
       verbose=2,
       callbacks=cbks
       )

print("complete training!")


#save weights
model.save_weights(weights_name, overwrite=True)
print("complete saving the weights")


finish_time = datetime.datetime.now()
finish_time_format =  '{0:%m%d_%H%M_finish}'.format(finish_time)
print(started_time_format)
print(finish_time_format)
delta = finish_time - started_time
print("time for training ")
print(delta)

# ### add for TensorBoard
KTF.set_session(old_session)
# ### end
