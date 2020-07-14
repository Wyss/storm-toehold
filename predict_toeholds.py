#Load up imports
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import keras as keras
from keras.models import load_model
from keras.regularizers import l2
from pysster.One_Hot_Encoder import One_Hot_Encoder
import pandas as pd
#from rq import get_current_job

from helper_gradient_ascent_functions import run_gradient_ascent, invert_onehot
from change_30nt_to_59nt import turn_switch_to_toehold, make_rev_complement
	
# create DNA alphabet
alph_letters = sorted('ATCG')
alph = list(alph_letters)
one = One_Hot_Encoder(alph_letters)

# one-hot encode with pysster (very fast and simple encoding)  
def _get_one_hot_encoding(seq):
    one_hot_seq = one.encode(seq)                         
    return one_hot_seq

def predict_seq(X, model_path, model_weights_path):
    model = load_model(model_path)
    model.load_weights(model_weights_path)
    predictions = model.predict(X)
    return predictions

def run_prediction(arg):
    seqs = [x.replace('U', 'T') for x in arg]
    X = np.stack([_get_one_hot_encoding(s) for s in seqs]).astype(np.float32)
    original_predictions = predict_seq(X, 'models/onoff_original_model.h5', 'models/onoff_original_model_weights.h5')
    transfer_predictions = predict_seq(X, 'models/freeze_weights_tf_onoff_model.h5', 'models/freeze_weights_tf_onoff_model.h5')

    orig_preds = [np.round(float(x), 3) for x in original_predictions]
    tf_preds = [np.round(float(x), 3) for x in transfer_predictions]
    
    data_df = pd.DataFrame(columns=['Toehold', 'Predicted ON/OFF (original)', 'Predicted ON/OFF (transfer)'])
    data_df['Toehold'] = seqs
    num_seqs = len(data_df)
    data_df['Predicted ON/OFF (original)'] = np.reshape(orig_preds, [num_seqs,])
    data_df['Predicted ON/OFF (transfer)'] = np.reshape(tf_preds, [num_seqs,])
    data_df = data_df.sort_values(by='Predicted ON/OFF (transfer)', ascending=False)
    pd.set_option('display.max_colwidth', -1)
    return data_df.to_html(justify='center', index=False)
    
def optimize_sequence(arg):
    seqs = [x.replace('U', 'T') for x in arg]
    X = np.stack([_get_one_hot_encoding(s) for s in seqs]).astype(np.float32)
    predictions = predict_seq(X, 'models/freeze_weights_tf_onoff_model.h5', 'models/freeze_weights_tf_onoff_model.h5')
    
    optimized_seqs = [] # store the converted sequences to be tested 
    predicted_targets = [] # store the original and predicted target values 

    num_rounds_of_optimization = 2
    num_seqs = 1

    for i in range(num_rounds_of_optimization):
        for idx, (toehold_seq, original_out) in enumerate(zip(seqs, predictions)): 
            optimized_pwm, optimized_onehot, predicted_out = run_gradient_ascent(toehold_seq, original_out)
            predicted_targets.append(predicted_out)
            new_seq = invert_onehot(np.reshape(optimized_onehot, [59,4]))
            optimized_seqs.append(new_seq)
    
    orig_preds = [np.round(float(x), 3) for x in predictions[0]]
    new_preds = [np.round(float(x[0][0]), 3) for x in predicted_targets]

    data_df = pd.DataFrame()
    data_df['Old Sequence'] = [seqs[0],seqs[0]]
    data_df['Old ON/OFF'] = [orig_preds[0], orig_preds[0]]
    data_df['new_switch'] = optimized_seqs
    data_df['predicted_onoff'] = new_preds

    # convert new switches to bp complementarity / toehold structure
    new_fixed_switches = []
    for toehold in data_df['new_switch']:
        base_30nt = toehold[0:30]
        new_toehold = turn_switch_to_toehold(base_30nt)
        new_fixed_switches.append(new_toehold)
    data_df['New Sequence'] = new_fixed_switches

    X = np.stack([_get_one_hot_encoding(s) for s in new_fixed_switches]).astype(np.float32)
    predictions = predict_seq(X, 'models/freeze_weights_tf_onoff_model.h5', 'models/freeze_weights_tf_onoff_model.h5')
    data_df['New ON/OFF'] = np.reshape(predictions, [num_seqs*num_rounds_of_optimization,])

    onoff_col = data_df.columns.get_loc("New ON/OFF")
    # cull so we have just the best out of each 5
    for i in range(0, num_seqs):
        start = i * num_rounds_of_optimization
        end = start + num_rounds_of_optimization
        best_toehold_so_far = data_df.iloc[start,:]
        for j in range(start+1, end):
            curr_toehold = data_df.iloc[j,:]
            if (data_df.iloc[j, onoff_col] > data_df.iloc[start, onoff_col]):
                best_toehold_so_far = curr_toehold
        best_seq = best_toehold_so_far['New Sequence']
        best_onoff = best_toehold_so_far['New ON/OFF']
    

    data_df = pd.DataFrame(columns=['Old Sequence', 'Old ON/OFF', 'New Sequence', 'New ON/OFF'])
    data_df['Old Sequence'] = [seqs[0]]
    data_df['Old ON/OFF'] = [orig_preds[0]]
    data_df['New Sequence'] = [best_seq]
    data_df['New ON/OFF'] = [best_onoff]

    pd.set_option('display.max_colwidth', -1)
    return data_df.to_html(justify='center', index=False)

def predict_genome_best(arg):
    long_genome = arg.replace('U', 'T')
    n = 30
    seqs = [(long_genome[i:i+n]) for i in range(0, len(long_genome)-n+1)] 
    seqs = [turn_switch_to_toehold(x) for x in seqs]

    X = np.stack([_get_one_hot_encoding(s) for s in seqs]).astype(np.float32)
    original_predictions = predict_seq(X, 'models/onoff_original_model.h5', 'models/onoff_original_model_weights.h5')
    transfer_predictions = predict_seq(X, 'models/freeze_weights_tf_onoff_model.h5', 'models/freeze_weights_tf_onoff_model.h5')

    orig_preds = [np.round(float(x), 3) for x in original_predictions]
    tf_preds = [np.round(float(x), 3) for x in transfer_predictions]
    
    data_df = pd.DataFrame(columns=['Toehold', 'Predicted ON/OFF (original)', 'Predicted ON/OFF (transfer)'])
    data_df['Toehold'] = seqs
    num_seqs = len(data_df)
    data_df['Predicted ON/OFF (original)'] = np.reshape(orig_preds, [num_seqs,])
    data_df['Predicted ON/OFF (transfer)'] = np.reshape(tf_preds, [num_seqs,])
    data_df = data_df.sort_values(by='Predicted ON/OFF (transfer)', ascending=False)
    pd.set_option('display.max_colwidth', -1)
    return data_df.to_html(justify='center', index=False)

