#Load up imports
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import keras as keras
from keras.models import load_model
from keras.regularizers import l2
from pysster.One_Hot_Encoder import One_Hot_Encoder

import isolearn.keras as iso
from seqprop import *
from seqprop.generator import *
from seqprop.predictor import *
from seqprop.optimizer import *

# define constants 

# get seed input which we will modify 
num_samples = 1

# template specifying what to modify and what not (biological constaints)
switch = 'NNNNNNNNNNNNNNNNNNNNNNNNNNNNNN'
rbs = 'AACAGAGGAGA'
start_codon = 'ATG'
stem1 = 'NNNNNN'#'XXXXXX'
stem2 = 'NNNNNNNNN'#'XXXXXXXXX'

bio_constraints = switch + rbs + stem1 + start_codon + stem2 

# define target on, off values 
target_onoff = 1
target = [[target_onoff], ]

final_model_path = 'models/freeze_weights_tf_onoff_model.h5'

alph_letters = sorted('ATCG')
alph = list(alph_letters)

# adapted from: https://github.com/876lkj/seqprop 

# need to re-create EXACT SAME layers as final trained model
# fix weights of layers so only input layer is modified
def load_saved_predictor(model_path) :

    saved_model = load_model(model_path)

    def _initialize_predictor_weights(predictor_model, saved_model=saved_model) :
        #Load pre-trained model
    
        predictor_model.get_layer('conv_0').set_weights(saved_model.get_layer('conv_0').get_weights())
        predictor_model.get_layer('conv_0').trainable = False

        predictor_model.get_layer('conv_1').set_weights(saved_model.get_layer('conv_1').get_weights())
        predictor_model.get_layer('conv_1').trainable = False

        predictor_model.get_layer('dense_0').set_weights(saved_model.get_layer('dense_0').get_weights())
        predictor_model.get_layer('dense_0').trainable = False

        predictor_model.get_layer('dense_1').set_weights(saved_model.get_layer('dense_1').get_weights())
        predictor_model.get_layer('dense_1').trainable = False

        predictor_model.get_layer('dense_2').set_weights(saved_model.get_layer('dense_2').get_weights())
        predictor_model.get_layer('dense_2').trainable = False

        predictor_model.get_layer('on_output').set_weights(saved_model.get_layer('on_output').get_weights())
        predictor_model.get_layer('on_output').trainable = False

    def _load_predictor_func(sequence_input) :
        # input space parameters 
        seq_length = 59
        num_letters = 4 # num nt 
        # expanded version b/c seqprop built for 2d 
        seq_input_shape = (seq_length, num_letters, 1) # modified

        #define new model definition (same architecture except modified input)
        dropout_rate=0.1
        reg_coeff= 0.0001
        hidden_layer_choices = {5: (150, 60, 15), }
        conv_layer_parameters = [(5,10), (3,5),]
        hidden_layers = hidden_layer_choices[5]
        
        #expanded_input = Input(shape=seq_input_shape,name='new_input')
        reshaped_input = Reshape(target_shape=(seq_length, num_letters),name='reshaped_input')(sequence_input)#(expanded_input)        #(kernel_width, num_filters) = conv_layer_parameters
        prior_layer = reshaped_input 
        for idx, (kernel_width, num_filters) in enumerate(conv_layer_parameters):
            conv_layer = Conv1D(filters=num_filters, kernel_size=kernel_width, padding='same', name='conv_'+str(idx))(prior_layer) # mimic a kmer
            prior_layer = conv_layer
        H = Flatten(name='flatten')(prior_layer)
        for idx,h in enumerate(hidden_layers): 
            H = Dropout(dropout_rate, name='dropout_'+str(idx))(H)
            H = Dense(h, activation='relu', kernel_regularizer=l2(reg_coeff), name='dense_'+str(idx))(H)
        out_onoff = Dense(1,activation="linear",name='on_output')(H)
        
        predictor_inputs = []
        predictor_outputs = [out_onoff]

        return predictor_inputs, predictor_outputs, _initialize_predictor_weights

    return _load_predictor_func
    
# build loss function
# ensure biological constraints are satisfied per sequence

def stem_base_pairing(pwm): 
    # ensure that location of 1s in switch region matches reverse complement of stem
    
    def reverse_complement(base_index): 
        # ACGT = alphabett
        if base_index == 0: return 3
        elif base_index == 1: return 2 
        elif base_index == 2: return 1 
        elif base_index == 3: return 0
    
    # reverse complement is reverse over axis of one-hot encoded nt 
    nt_reversed = K.reverse(pwm, axes=2)
    stem1_score = 6 - K.sum(pwm[:, 24, :, 0]*nt_reversed[:, 41,:, 0] + pwm[:, 25, :, 0]*nt_reversed[:, 42, :, 0]+ pwm[:,26, :, 0]*nt_reversed[:, 43, :, 0] + pwm[:, 27, :, 0]*nt_reversed[:, 44, :, 0] + pwm[:, 28, :, 0]*nt_reversed[:, 45, :, 0]+ pwm[:, 29, :, 0]*nt_reversed[:, 46, :, 0])
    stem2_score = 9 - K.sum(pwm[:, 12, :, 0]*nt_reversed[:, 50, :, 0] + pwm[:, 13, :, 0]*nt_reversed[:, 51, :, 0]+ pwm[:, 14, :, 0]*nt_reversed[:, 52, :, 0]+ pwm[:, 15, :, 0]*nt_reversed[:, 53, :, 0] + pwm[:, 16, :, 0]*nt_reversed[:, 54, :, 0] + pwm[:, 17, :, 0]*nt_reversed[:,55, :, 0]+ pwm[:, 18,:, 0]*nt_reversed[:, 56, :, 0] + pwm[:, 19, :, 0]*nt_reversed[:,57, :, 0] + pwm[:, 20, :, 0]*nt_reversed[:, 58, :, 0])
    return 10*stem1_score + 10*stem2_score

def loss_func(predictor_outputs) :
    pwm_logits, pwm, sampled_pwm, predicted_out = predictor_outputs
  
    #Create target constant -- want predicted value for modified input to be close to target input 
    target_out = K.tile(K.constant(target), (K.shape(sampled_pwm)[0], 1))
    target_cost = (target_out - predicted_out)**2
    print(target_out, target_cost, predicted_out)
    base_pairing_cost = stem_base_pairing(sampled_pwm)
    print(base_pairing_cost)
    print(K.mean(target_cost + base_pairing_cost, axis=-1))
    #return K.mean(target_cost + base_pairing_cost, axis=-1)
    return K.mean(target_cost, axis=-1)
    
def run_gradient_ascent(input_toehold_seq, original_out):
    seq_length = 59
    # build generator network
    _, seqprop_generator = build_generator(seq_length=seq_length, n_sequences=num_samples, batch_normalize_pwm=True,init_sequences = [input_toehold_seq],
                                          sequence_templates=bio_constraints)# batch_normalize_pwm=True)
    
    # build predictor network and hook it on the generator PWM output tensor
    _, seqprop_predictor = build_predictor(seqprop_generator, load_saved_predictor(final_model_path), n_sequences=num_samples, eval_mode='pwm')

    #Build Loss Model (In: Generator seed, Out: Loss function)
    _, loss_model = build_loss_model(seqprop_predictor, loss_func, )

    #Specify Optimizer to use
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    #Compile Loss Model (Minimize self)
    loss_model.compile(loss=lambda true, pred: pred, optimizer=opt)

    #Fit Loss Model
    #seed_input = np.reshape([X[0]], [1,59,4,1]) # any input toehold to be modified

    callbacks =[
                EarlyStopping(monitor='loss', min_delta=0.001, patience=5, verbose=0, mode='auto'),
                #SeqPropMonitor(predictor=seqprop_predictor)#, plot_every_epoch=True, track_every_step=True, )#cse_start_pos=70, isoform_start=target_cut, isoform_end=target_cut+1, pwm_start=70-40, pwm_end=76+50, sequence_template=sequence_template, plot_pwm_indices=[0])
            ]


    num_epochs=50
    train_history = loss_model.fit([], np.ones((1, 1)), epochs=num_epochs, steps_per_epoch=1000, callbacks=callbacks)

    #Retrieve optimized PWMs and predicted (optimized) target
    _, optimized_pwm, optimized_onehot, predicted_out = seqprop_predictor.predict(x=None, steps=1)
    print('Original [on/off]:', original_out)
    print('Predicted [on/off]: ', predicted_out)
    
    return optimized_pwm, optimized_onehot, predicted_out
    
def invert_onehot(oh_seq): 
    return ''.join(alph[idx] for idx in np.argmax(oh_seq,axis=1))