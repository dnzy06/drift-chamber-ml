from datetime import datetime

def print_flush(*args, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = ' '.join(str(arg) for arg in args)
    
    # Write to both stdout AND a dedicated file
    print(f"[{timestamp}] {message}", flush=True)
    
    # Also write directly to file (this will definitely appear)
    with open('hyperparameter_search_direct.log', 'a') as f:
        f.write(f"[{timestamp}] {message}\n")
        f.flush()

print_flush("Starting script...")

import numpy as np
from tqdm import tqdm

compress = False
compress_factor = 1

if compress == False:
    compress_factor = 1

truncate = True
num_timepoints = 500

if truncate == False:
    num_timepoints = 3000

batchSize = 64

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

            # print(y)
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

# Load file paths
file_list = sorted(glob.glob("./data/processed_data_train/*.npz"))
train_files, val_files = train_test_split(file_list, test_size=0.2, shuffle=False)

train_ds = build_dataset(train_files, batchSize)
val_ds = build_dataset(val_files, batchSize, shuffle=False)

import keras
import tensorflow as tf
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from qkeras import *
import numpy as np
import pandas as pd
from itertools import product
import json

class QuantizationSearcher:
    def __init__(self, train_ds, val_ds, num_timepoints, compress_factor):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.num_timepoints = num_timepoints
        self.compress_factor = compress_factor
        self.results = []
        
    def create_model(self, config):
        """Create quantized model with given configuration"""
        # Create quantizers
        input_quantizer = quantized_bits(16, 10)
        kernel_bias_quantizer = quantized_bits(config['total_bits'], config['weight_integer_bits'], symmetric = 1)
        relu_quantizer = quantized_relu(config['total_bits'], config['relu_integer_bits'])
        linear_quantizer = quantized_bits(config['total_bits'], config['linear_integer_bits'])

        # Build model
        inputs = Input(shape=(int(self.num_timepoints/self.compress_factor),))
        x = QActivation(input_quantizer)(inputs)
        
        x = QDense(8, kernel_quantizer=kernel_bias_quantizer, bias_quantizer=kernel_bias_quantizer)(x)
        x = QActivation(relu_quantizer)(x)
        
        x = QDense(32, kernel_quantizer=kernel_bias_quantizer, bias_quantizer=kernel_bias_quantizer)(x)
        x = QActivation(relu_quantizer)(x)
        
        x = QDense(8, kernel_quantizer=kernel_bias_quantizer, bias_quantizer=kernel_bias_quantizer)(x)
        x = QActivation(relu_quantizer)(x)
        
        x = QDense(1, kernel_quantizer=kernel_bias_quantizer, bias_quantizer=kernel_bias_quantizer)(x)
        outputs = QActivation(linear_quantizer)(x)
        
        return keras.Model(inputs=inputs, outputs=outputs, name="DNN")
    
    def train_evaluate(self, config):
        """Train and evaluate model with given configuration"""
        try:
            # Create and compile model
            model = self.create_model(config)
            
            loss_fn = tf.keras.losses.MeanSquaredError()
            opt = Adam(learning_rate=config['learning_rate'])
            model.compile(loss=loss_fn, optimizer=opt, 
                         metrics=tf.keras.metrics.MeanAbsoluteError(name='mae'))
            
            # Callbacks
            earlystop = EarlyStopping(monitor="val_loss", patience=10, 
                                    restore_best_weights=True, min_delta=0.0)
            reducelr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, 
                                       patience=3, min_delta=0., min_lr=1e-7)
            
            # Train for 10 epochs
            history = model.fit(self.train_ds, validation_data=self.val_ds,
                              epochs=10, callbacks=[earlystop, reducelr], verbose=0)
            
            # Evaluate on validation set
            val_loss, val_mae = model.evaluate(self.val_ds, verbose=0)
            
            return {
                'val_loss': val_loss,
                'val_mae': val_mae,
                'success': True,
                'total_bits': config['total_bits']
            }
            
        except Exception as e:
            return {
                'val_loss': float('inf'),
                'val_mae': float('inf'),
                'total_bits': 64,
                'success': False,
                'error': str(e)
            }
    
    def search(self, aggressive=True):
        """Run complete grid search"""
        
        # Define search space
        space = {
            'total_bits': [6, 8],
            'weight_integer_bits': [2, 3, 4, 5],
            'relu_integer_bits': [2, 3, 4, 5],
            'linear_integer_bits': [2, 3, 4, 5],
            'learning_rate': [1e-4, 3e-4, 5e-4, 6e-4]
        }
        
        # Generate all combinations
        param_names = list(space.keys())
        param_values = list(space.values())
        all_combinations = list(product(*param_values))
        
        total_trials = len(all_combinations)
        print_flush(f"Starting complete grid search with {total_trials} combinations (aggressive={aggressive})")
        
        for trial, combination in enumerate(all_combinations):
            # Create configuration dictionary
            config = dict(zip(param_names, combination))
            
            print_flush(f"Trial {trial+1}/{total_trials}: "
                        f"total bits=({config['total_bits']}) "
                  f"bits=({config['weight_integer_bits']},"
                  f"{config['relu_integer_bits']},{config['linear_integer_bits']}) "
                  f"lr={config['learning_rate']:.0e}")
            
            # Train and evaluate
            result = self.train_evaluate(config)
            result.update(config)
            result['trial'] = trial + 1
            self.results.append(result)
            
            if result['success']:
                print_flush(f"  Loss: {result['val_loss']:.4f}, "
                      f"Bits: {result['total_bits']}")
            else:
                print_flush(f"  FAILED: {result.get('error', 'Unknown error')}")
        
        return self.get_best_configs()
    
    import pandas as pd

    def get_best_configs(self):
        """Get top performing configurations from each 'total_bits' group."""
        successful = [r for r in self.results if r['success']]
        if not successful:
            print_flush("No successful configurations found!")
            return []

        # Convert results to a DataFrame for easier grouping
        df = pd.DataFrame(successful)
        
        # Group by 'total_bits' and find the best config in each group
        best_by_bits = df.loc[df.groupby('total_bits')['val_loss'].idxmin()]
        
        # Sort the best configurations by their total bits value for a clean output
        best_by_bits = best_by_bits.sort_values('total_bits').to_dict('records')

        print_flush(f"\n{'='*60}")
        print_flush("TOP CONFIGURATIONS BY TOTAL BITS")
        print_flush(f"{'='*60}")
        
        for i, config in enumerate(best_by_bits):
            print_flush(f"\nTotal Bits: {config['total_bits']}")
            print_flush(f"  Val Loss: {config['val_loss']:.4f}")
            print_flush(f"  Val MAE: {config['val_mae']:.4f}")
            print_flush(f"  Config: total_bits({config['total_bits']}) "
                f"weight({config['weight_integer_bits']},) "
                f"relu({config['relu_integer_bits']}) "
                f"linear({config['linear_integer_bits']},) "
                f"lr={config['learning_rate']:.0e}")
            
        return best_by_bits
        
    
    def save_results(self, filename="quantization_results"):
        """Save results to files"""
        df = pd.DataFrame(self.results)
        df.to_csv(f"{filename}.csv", index=False)
        
        with open(f"{filename}.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print_flush(f"\nResults saved to {filename}.csv and {filename}.json")


def find_best_quantization(train_ds, val_ds, num_timepoints, compress_factor, 
                          aggressive=True):
    """
    Main function to find optimal quantization configuration using grid search
    
    Args:
        train_ds: TensorFlow training dataset
        val_ds: TensorFlow validation dataset  
        num_timepoints: Model input dimension parameter
        compress_factor: Model input dimension parameter
        aggressive: If True, focuses on lower bit configurations
    
    Returns:
        List of best configurations
    """
    
    searcher = QuantizationSearcher(train_ds, val_ds, num_timepoints, compress_factor)
    best_configs = searcher.search(aggressive=aggressive)
    searcher.save_results()
    
    return best_configs


# Usage example:

# Run the grid search
best_configs = find_best_quantization(
    train_ds=train_ds,
    val_ds=val_ds, 
    num_timepoints=num_timepoints,
    compress_factor=compress_factor,
    aggressive=True
)

best_configs.to_csv("best_quantization_configs.csv", index=False)
print(best_configs)
