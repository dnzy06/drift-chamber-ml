data_dir = '../data/'
model_dir = '../models/'
results_dir = '../results/'

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

def make_corr_plot(actual, predicted):

    m = 1
    b = 0
    correlation_coefficient, p_value = pearsonr(actual[:10000], predicted[:10000])
    x = np.arange(actual.min(), actual.max(), step=0.1)

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, s=30, alpha=0.2)
    plt.plot(x, m*x + b, color='red')
    plt.xlabel('Truth Number of Primary Ionizations')
    plt.ylabel('Predicted Number of Primary Ionizations')
    plt.text(0.05,0.95, f"Correlation coefficient = {correlation_coefficient:.2f}", fontsize=12, color='black', 
             transform=plt.gca().transAxes,
             horizontalalignment='left',    
             verticalalignment='top', )
    plt.grid(True, color='gray', linestyle='--', alpha=0.4)
    plt.title('Truth vs. Predicted Number of Primary Ionizations')

def plot_percent_error(actual, predicted):
    mask = actual != 0
    actual = actual[mask]
    predicted = predicted[mask]
    error = ((predicted - actual)/actual) * 100

    plt.hist(error,bins=np.arange(-100, 100, 2),histtype='bar',label='Error', edgecolor="black", fc="#69b3a2", alpha=0.3)
    plt.title('Histogram of Percent Error')
    plt.xlabel('Error (%)')

    mean = np.mean(error)
    std = np.std(error.astype(np.float32))

    plt.axvline(mean, color='red', linestyle='--', linewidth=2)
    
    plt.text(0.05, 0.95, f'Mean: {mean:.2f}\nSTD: {std:.2f}',
    transform=plt.gca().transAxes,
    horizontalalignment='left',    
    verticalalignment='top',  
    color='black', fontsize=11)
    plt.grid(True, color='gray', linestyle='--', alpha=0.4)


def flatten_list(item, flattened):

    if isinstance(item, list) or isinstance(item, np.ndarray):
        for sub_item in item:
            flatten_list(sub_item, flattened)
    
    else:
        flattened.append(item)

def box_plot_weights(model):
    layers = model.layers
    layer_weights = []
    layer_names = []

    for layer in layers:
        if (len(layer.get_weights()) != 0):
            layer_names.append(layer.name)
            curr_weights = []
            flatten_list(layer.get_weights(), curr_weights)
            layer_weights.append(curr_weights)
            
    print(layer_names)
    print(layer_weights)

    print(len(layer_names))
    print(len(layer_weights))

    fig, ax = plt.subplots()
    bp = ax.boxplot(layer_weights, vert = False, patch_artist=True, labels = layer_names)

    from matplotlib import cm
    from matplotlib.colors import Normalize

    cmap = cm.Blues
    # Normalize values for the colormap (e.g., from 0 to 1 for a smooth gradient)
    norm = Normalize(vmin=0, vmax=1) 

    for i, box in enumerate(bp['boxes']):
            color = cmap(norm(i / (len(data) - 1))) # Normalize index to 0-1 range
            box.set_facecolor(color)

    yticks = []

    for i in range(len(layer_names)):
        yticks.append(i + 1)
    plt.yticks(yticks, layer_names)
    plt.xlabel('Weight Values')
    plt.title('Weight Distribution')

    plt.show()


compress = False
compress_factor = 1

if compress == False:
    compress_factor = 1

truncate = True
num_timepoints = 500

if truncate == False:
    num_timepoints = 3000

def load_data(dir, files):
    # Dictionary to store lists of arrays for each key
    combined_data = {}

    # Loop through files and collect arrays
    for filename in files:
        origdata = np.load(dir+filename)
        for key in origdata.files:
            if key not in combined_data:
                combined_data[key] = []
            combined_data[key].append(origdata[key])

    # Concatenate arrays along the first axis for each key
    return {key: np.concatenate(arr_list, axis=0) for key, arr_list in combined_data.items()}


data = load_data(data_dir + "processed_data_train/",
                 ["batch_0.npz",
                  #"batch_1.npz",
                  #"batch_2.npz",
                  #"batch_3.npz",
                  #"batch_4.npz", 
                 ])
data = {d:data[d][:100000] for d in data} # reduce size in memory

if truncate:
    data['wf_i'] = data['wf_i'][:,:num_timepoints]
    mask = data['tag_times'] >= num_timepoints
    data['tag_times'][mask] = 0
    data['tag_values'][mask] = 0

testdata = load_data(data_dir + "processed_data_test/",
                     ["batch_0.npz",
                      #"batch_1.npz",
                      #"batch_2.npz",
                      #"batch_3.npz",
                      #"batch_4.npz",
                    ])

if compress:
    compressed = []
    for i in range(testdata['wf_i'].shape[0]):
        wf_i = testdata['wf_i'][i]
        wf_i = wf_i.reshape(-1, compress_factor)
        wf_i = wf_i.mean(axis=1)

        compressed.append(wf_i)

    testdata['wf_i'] = np.asarray(compressed)

if truncate:
    testdata['wf_i'] = testdata['wf_i'][:,:num_timepoints]
    mask = testdata['tag_times'] >= num_timepoints
    testdata['tag_times'][mask] = 0
    testdata['tag_values'][mask] = 0
        

def expand_values(values, times, tlength=3000, ohe=True, combine=False):
    # values: (300,)
    # times: (300,)
    data = np.zeros((num_timepoints,), dtype=np.float32)

    # Place values at the correct times
    valid_mask = (times >= 0) & (times < tlength)
    times = times[valid_mask]
    values = values[valid_mask]

    if truncate:
        mask = times >= num_timepoints
        times[mask] = 0
        values[mask] = 0

    data[times] = values
    
    if ohe:
        ohe_data = np.zeros((num_timepoints, 3), dtype=np.float32)
        for i in range(3):
            ohe_data[:, i] = (data == i).astype(np.float32)
        if combine:
            ohe_data[:, 1] = np.any(ohe_data[:, 1:], axis=-1)
            ohe_data[:, 2] = 0.0
        return ohe_data
    else:
        if combine:
            data = (data > 0).astype(np.float32)
        return data[:, None]  # shape (3000, 1)
    
def np_expand_values(values,times,tlength = 3000, ohe = True, combine = False):
    data = np.zeros((values.shape[0],tlength))
    data[np.arange(times.shape[0])[:,None],times] = values
    if ohe:
        data = np.concatenate([(data==i)[:,:,None] for i in range(3)],axis=-1).astype(np.float32)
        if combine:
            data[:,:,1] = np.any(data[:,:,1:],axis=-1)
            data[:,:,2] = 0.
    else:
        if combine:
            data = (data>0).astype(np.float32)
    return data

def batched_expand_values(values, times, batch_size=1000, tlength=3000, ohe=True, combine=False):
    outputs = []
    for i in tqdm(range(0, len(values), batch_size)):
        v_batch = values[i:i+batch_size]
        t_batch = times[i:i+batch_size]
        out = np_expand_values(v_batch, t_batch, tlength=tlength, ohe=ohe, combine=combine)
        outputs.append(out)
    return np.concatenate(outputs, axis=0)

import tensorflow as tf
from tensorflow import keras
# Import tf/keras modules
from keras.layers import Dense
from keras.layers import Input
# This clears the Keras session
tf.keras.backend.clear_session()

import matplotlib.pyplot as plt

import os, sys

def load_npz_individual(file_path_bytes):
    path = file_path_bytes.decode("utf-8")
    data = np.load(path)

    # print(data)
    wf_i = data["wf_i"][:, :num_timepoints].astype(np.float32)      # (N, 3000)
    # print('wf shape ', wf_i.shape)
    tag_values = data["tag_values"]               # (N, ?)
    tag_times = data["tag_times"]                 # (N, ?)

    if truncate:
        mask = tag_times >= num_timepoints
        tag_times[mask] = 0
        tag_values[mask] = 0

    return wf_i, tag_values, tag_times

def build_dataset(file_list, batch_size, tlength=3000, ohe=True, combine=False, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(file_list)

    def file_to_dataset(file_path):
        wf_i, tag_values, tag_times = tf.numpy_function(
            func=load_npz_individual,
            inp=[file_path],
            Tout=[tf.float32, tf.int32, tf.int32]
        )
        # Shapes are unknown, need to set
        wf_i.set_shape([None, num_timepoints])         # (N, 3000)

        if compress:
            wf_i = tf.reshape(wf_i, [tf.shape(wf_i)[0], -1, compress_factor])
            wf_i = tf.reduce_mean(wf_i, axis=-1)
            wf_i.set_shape([None, int(num_timepoints/compress_factor)]) 
  
        tag_values.set_shape([None, 300])     # (N, ?)
        tag_times.set_shape([None, 300])       # (N, ?)

        ex_ds = tf.data.Dataset.from_tensor_slices((wf_i, tag_values, tag_times))

        def process(wf_i_ex, tag_values_ex, tag_times_ex):
            # print(wf_i_ex)
            wf_i_ex = tf.expand_dims(wf_i_ex, axis=-1)  # (3000, 1)
            # print(tag_values_ex)
            boolean_tensor = tf.equal(tag_values_ex, 1)
            # print('boolean tensor ', boolean_tensor)
            y = tf.reduce_sum(tf.cast(boolean_tensor, tf.int32))

            print(y)
            if compress:
                wf_i_ex.set_shape([int(num_timepoints/compress_factor), 1])

            else:
                wf_i_ex.set_shape([num_timepoints, 1])
                
            return wf_i_ex, y
        

        return ex_ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)

    #if shuffle:
    #    ds = ds.shuffle(len(file_list))

    ds = ds.flat_map(file_to_dataset)
    ds = ds.shuffle(512)  # shuffle individual examples
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

from sklearn.model_selection import train_test_split
import glob

batchSize = 64

# Load file paths
file_list = sorted(glob.glob(data_dir + "processed_data_train/*.npz"))
train_files, val_files = train_test_split(file_list, test_size=0.2, shuffle=False)

train_ds = build_dataset(train_files, batchSize)
val_ds = build_dataset(val_files, batchSize, shuffle=False)

import keras
from keras.layers import Dense, Input, AveragePooling1D, Reshape, GlobalAveragePooling1D, MaxPooling1D, Flatten

# seeing by how much we can quantize the input layer

from qkeras import * 

input_quantizer = quantized_bits(16, 4, symmetric=0)
input_quantizer(testdata['wf_i'][0]).numpy()

import keras
from keras.layers import Dense, Input, AveragePooling1D, Reshape, GlobalAveragePooling1D, MaxPooling1D, Flatten
from qkeras import *

kernel_bias_quantizer = quantized_bits(10, 5, symmetric=1)
relu_quantizer = quantized_relu(10, 5)
linear_quantizer = quantized_bits(10, 5)
input_quantizer = quantized_bits(10, 5)

tf.random.set_seed(13)
np.random.seed(13)

from qkeras import * 

from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects

co = {}
_add_supported_quantized_objects(co)
model = load_model(model_dir + 'hls4ml/10_5_500_dense_8_32_8_lr=3e-4.h5', custom_objects=co)

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.optimizers import Adam

import tempfile

epochs = 7
learningRate = 6e-4
batchSize = 64
loss_fn = tf.keras.losses.MeanSquaredError()

sparsity_values = [0.2, 0.4, 0.6, 0.8, 0.99]

def pruning_scan(sparsity_values, model):

    model_paths = []
    mean_error = []
    std_error = []
    for sparsity in sparsity_values:
        print("============================== SPARSITY: " + str(sparsity) + "==============================")
        model_for_pruning = model

        pruning_params = {"pruning_schedule": pruning_schedule.ConstantSparsity(sparsity, begin_step=0, frequency=100)}
        model_for_pruning = prune.prune_low_magnitude(model, **pruning_params)

        # `prune_low_magnitude` requires a recompile.
        opt = Adam(learning_rate=learningRate)
        model_for_pruning.compile(optimizer=opt,
                    loss=loss_fn,
                    metrics=tf.keras.metrics.MeanAbsoluteError(name='mae'))

        # model_for_pruning.summary()

        logdir = tempfile.mkdtemp()

        callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
        ]

        history = model_for_pruning.fit(train_ds, validation_data=val_ds,
                            batch_size=batchSize, epochs=epochs, callbacks=callbacks)

        model_for_pruning = strip_pruning(model_for_pruning)

        

        preds = model_for_pruning.predict(testdata['wf_i'])
        actual = np.sum(testdata['tag_values'] == 1, axis = 1)

        preds = np.asarray(preds.astype(np.float16))[:1000].reshape(1000)
        actual = np.asarray(actual.astype(np.float16))[:1000].reshape(1000)

        mask = actual != 0
        actual = actual[mask]
        preds = preds[mask]

        error = ((preds - actual)/actual) * 100

        mean = np.mean(error)
        std = np.std(error.astype(np.float32))

    return mean_error, std_error



mean_error, std_error = pruning_scan(sparsity_values, model)


import matplotlib.pyplot as plt
def plot_sparsity_against_mean_error(sparsity_values, model_means):
    plt.errorbar(np.sort(sparsity_values), np.asarray(model_means)[np.argsort(sparsity_values)], fmt='o-')
    plt.xlabel("Sparsity")
    plt.ylabel("Mean Percent Error")
    plt.grid(True, color='gray', linestyle='--', alpha=0.4)
    plt.title("Sparsity vs. Mean Percent Error")
    plt.savefig(results_dir + 'sparsity_vs_mean_error.png')

def plot_sparsity_against_std_error(sparsity_values, std_means):
    plt.errorbar(np.sort(sparsity_values), np.asarray(std_means)[np.argsort(sparsity_values)], fmt='o-')
    plt.xlabel("Sparsity")
    plt.ylabel("STD Percent Error")
    plt.grid(True, color='gray', linestyle='--', alpha=0.4)
    plt.title("Sparsity vs. STD Percent Error")
    plt.savefig(results_dir + 'sparsity_vs_std_error.png')

plot_sparsity_against_mean_error(sparsity_values, mean_error)
plot_sparsity_against_std_error(sparsity_values, std_error)