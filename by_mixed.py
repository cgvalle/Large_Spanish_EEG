# Effects of amount of data
from src.models import EEGNet, DeepConvNet, ShallowConvNet
from src.dataloader import SpeechDataset
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras import utils as np_utils
import numpy as np
import parameters as p
import argparse
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description='Experiments by condition (rest, perception and production)')
parser.add_argument('--num_runs', type=int, default=30, help="Number of repetitions")
parser.add_argument('--window', type=int, default=750, help="Window length in samples")
parser.add_argument('--task', type=str, help="Task of dataset (perception or production)")
parser.add_argument('--stims', type=int, default=5, help="Num of stims")
parser.add_argument('--shuffle', type=bool, default=False)
parser.add_argument('--name', type=str, default='normal')

args = parser.parse_args()


STIMS = list(range(1,31)[:args.stims])


if not os.path.exists(f"results/mixed/{args.task}/{args.name}/stims_{args.stims}"):
    os.makedirs(f"results/mixed/{args.task}/{args.name}/stims_{args.stims}")



P = []
for iteration in tqdm(range(args.num_runs)):

    ### Load data ###
    speech_dataset = SpeechDataset(p.DATASET, 
                        window=args.window, 
                        shuffle_labels = args.shuffle, 
                        stims=STIMS, 
                        tasks=[args.task], 
                        subjects=p.SUBJECTS)
    X_train, Y_train, X_test, Y_test = speech_dataset.balanced_split(takes=1, shuffle_labels=args.shuffle)






    
    # One-hot encoding of labels
    Y_train = np_utils.to_categorical(Y_train-1)
    Y_test = np_utils.to_categorical(Y_test-1)

    ### model ###
    ### model ###
    if args.name == 'eegnet':
        model = EEGNet(len(STIMS), Chans = 14, Samples = args.window, 
                        dropoutRate = 0.25, kernLength = 125, F1 = 8, 
                        D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout')
    
    if args.name == 'deepconv':
        model = DeepConvNet(len(STIMS), Chans=14, Samples=args.window, dropoutRate=0.5)

    if args.name == 'shallow':
        model = ShallowConvNet(len(STIMS), Chans = 14, Samples = args.window, dropoutRate = 0.5)
    model.summary()


    loss = 'categorical_crossentropy'
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                    metrics = ['accuracy'])


    ### Training ###
    # Early stopping
    early_stopping_callbcak = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=5
    )


    history = model.fit(X_train, Y_train, 
                        batch_size = 32, 
                        epochs = 200, 
                        verbose = 1, 
                        validation_data=(X_test, Y_test), 
                        callbacks=[early_stopping_callbcak])

    P.append(history.history['val_accuracy'][-5])
    
    np.save(f'results/mixed/{args.task}/{args.name}/stims_{args.stims}/task_{args.task}_shuffle_{args.shuffle}', P) 
    

