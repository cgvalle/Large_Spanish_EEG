import mne
import os; os.system('clear')
from matplotlib import pyplot as plt
import json
import numpy as np
from collections import namedtuple
from mne.preprocessing import ICA
from tqdm import tqdm
import argparse
from mne_icalabel import label_components
from main_pipeline import load_subject, load_stimulus, add_montage, pre_processing

parser = argparse.ArgumentParser(description='subjects')

parser.add_argument('-subject',type=str, default='sub-038', help="subject")
parser.add_argument('-session',type=str, default='ses-01', help="session")

args = parser.parse_args()



    
# Step 5. pre-processing
def pre_processing(raw, l_freq, h_freq, reference='average', only_language=True):
    # Load data
    raw.load_data()

    # Drop bad Channels
    raw.drop_channels(['EMG', 'EKG'])
    
    # Filte data
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    
    
    # Re-reference data
    if reference is not None:
        if reference == 'average':
            raw.set_eeg_reference('average', ch_type='eeg')
        else:
            raw.set_eeg_reference(reference, ch_type='eeg')
            
    # Interpolate bad channels
    #raw.interpolate_bads()

    # Pick language channels
    if only_language:
        raw.pick(['F7', 'F5', 'F3', 'FT7', 'FC5', 'T7', 'C5', 'C3', 'TP7', 'CP5', 'CP3', 'P7', 'P5', 'P3'])
    
    return raw



if __name__ == '__main__':
    # General Variables
    DATASET_FOLDER_PATH = 'ds004279-download'  # Here all the subjects are stored
    
    # Step 0. Load aux info
    index2stim = load_stimulus()

    l_freq = 1
    h_freq = 50

    subject = args.subject
    session = args.session

    # Step 1. Load data
    raw = load_subject(subject, session, DATASET_FOLDER_PATH, preload=True)  # triggers_sequencial

    # Step 2. Add montage to data
    raw = add_montage(raw)

    
    ica_raw = raw.copy()
    ica_raw.load_data()
    ica_raw.filter(l_freq=1, h_freq=100)

    # Step 5. Pre-process data
    raw = pre_processing(raw, l_freq=l_freq, h_freq=h_freq, reference='average', only_language=False)

    # Step 6. Run ICA
    n_components = 45 
    ica = ICA(n_components=n_components, max_iter='auto', random_state=42, method='infomax', fit_params=dict(extended=True))
    ica.fit(ica_raw)

    ica.save(f'ica/{subject}_{session}_task-sentences_icaWeights.fif', overwrite=True)

        
    ic_labels = label_components(raw, ica, method='iclabel')
    print(type(ic_labels))
    print(ic_labels)

    tolist_ic_labels = {
        'y_pred_proba': ic_labels['y_pred_proba'].tolist(), 
        'labels': ic_labels['labels'],
    }
    with open(os.path.join('ica', f'{subject}_{session}_task-sentences_icaLabels.json'), 'w') as f:
        json.dump(tolist_ic_labels, f)
