import os; os.system('clear')
import mne
import pandas as pd
from matplotlib import pyplot as plt
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from mne.preprocessing import read_ica




## Load stimulus dict
def load_stimulus():
    # Load stimulus from stimulus_dict.json
    # The key is the index of the stimulus and the value is the text
    with open('stimulus_dict.json', 'r') as stimulus_file:
        index2stim = json.load(stimulus_file)
    return index2stim


## Step 1. Data Loading
def load_subject(subject_name, session, DATA_FOLDER_PATH, preload=False):
    # Paths for the subject
    edf_file_path = f"{subject_name}/{session}/eeg/{subject_name}_{session}_task-sentences_eeg.edf"
    edf_full_path = os.path.join(DATA_FOLDER_PATH, edf_file_path)
    
    #################################
    # Load raw data into mne format #
    #################################
    raw = mne.io.read_raw_edf(edf_full_path, stim_channel="Trigger", preload=preload, verbose=False)
    

    ####################
    #### Load Events ###
    ####################
    events_file_path = f"{subject_name}/{session}/eeg/{subject_name}_{session}_task-sentences_events.tsv"
    events = pd.read_csv(os.path.join(DATA_FOLDER_PATH, events_file_path), sep='\t')
    annotations = mne.Annotations(onset=events['onset'], 
                                  duration=events['duration'], 
                                  description=events['trial_type'], 
                                  orig_time=None)
    raw.set_annotations(annotations)
    
    
    return raw
    
    
    
    
# Step 2. Add Montage to data
def add_montage(raw):
    # A montage is needed to perform interpolations between channels and ica

    # Set channels type
    raw.set_channel_types({'HEO':'eog', 'VEO':'eog', 'EKG':'ecg', 'EMG':'emg', 'Trigger':'stim'})

    # Load standar montage 10-05
    montage_1005 = mne.channels.make_standard_montage('standard_1005')
    
    # Change channels name to match with the ones in the montage
    old2new = {'FP1':'Fp1', 'FPZ':'Fpz','FP2': 'Fp2' , 
                'FZ': 'Fz', 'FCZ':'FCz', 'CZ':'Cz', 'CPZ':'CPz',
                'PZ':'Pz', 'POZ':'POz', 'OZ':'Oz', 'CB1':'I1', 'CB2':'I2'}
    mne.rename_channels(raw.info, old2new)
    
    # Set montage
    raw.set_montage(montage_1005, on_missing='raise')
    
    # Optional plot for channel location visualization
    if False:
        raw.plot_sensors(show_names=True, kind='3d')
        plt.show()
    
    return raw
    

        

    
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
            

    # Pick language channels
    if only_language:
       raw.pick(['F7', 'F5', 'F3', 'FT7', 'FC5', 'T7', 'C5', 'C3', 'TP7', 'CP5', 'CP3', 'P7', 'P5', 'P3'])
    


    return raw

# Step 6. Crete epochs
def create_epochs(raw, EPOCHS, subject, session):
    # Creates epochs for each perception, production, rest and preparation
    
    # Bad events are automatically dropped
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    
    # Get duration of each condition
    condition_duration = defaultdict(list)
    for annot in raw.annotations:
        condition_duration[annot['description'].split('_')[0]].append(annot['duration'])
    
    
    subject = int(subject.replace('sub-', ''))
    session = int(session.replace('ses-', ''))
    
    
    for condition in ['rest', 'perception', 'preparation', 'production']:   
        
        assert len(set(condition_duration[condition])) == 1, "Error on the duration"
        
        
        
        for stim in range(1, 31):
            temp_epoch = mne.Epochs(raw, 
                            events=events,
                            event_id=event_id[f'{condition}_{stim}'],
                            tmin=0,      # if -1 : extended
                            tmax=condition_duration[condition][0] - 0.001,
                            preload=False,
                            on_missing='raise',
                            reject_by_annotation=True,
                            baseline=None, 
                            verbose=False
                            )
            temp_epoch.drop_bad()
            temp_epoch.load_data()
            temp_epoch.resample(250, verbose=True)

            # Subjects with multiple sessions get a letter to indicate the session
            if subject == 57 or subject == 63:
                subject = f"{subject}{'abcd'[session-1]}"

            
            EPOCHS[f"S{subject}_{condition}_{stim}"] = temp_epoch
    
    return EPOCHS
    
    
def save_data(EPOCHS, filename):
    loaded_epochs = {}
    for key, epoch in EPOCHS.items():
        loaded_epochs[key] = epoch.get_data(units='uV', picks='eeg')
    # TODO: Darle una mirada a los diferentes parametros de get_data
    
    # Verify that the dtype of each array is float64
    for array in loaded_epochs.values():
        assert array.dtype == np.float64, "Bad dtype"
    
    np.savez(filename, **loaded_epochs)
    
    
def remove_ica_componenets(raw, subject, session, rejection_coinfidence=0.95):
    ica_label_path = f"ica/{subject}_{session}_task-sentences_icaLabels.json"
    with open(ica_label_path) as f:
        ica_dict = json.load(f)
    
    exclude_components = []
    for index, (label, confidence) in enumerate(zip(ica_dict['labels'], ica_dict['y_pred_proba'])):
        # Do not remove brain signals
        if label == 'brain':
            print(index, f" {confidence:.2f} {label} {'+'*20}")
            continue

        if confidence >= rejection_coinfidence:
            exclude_components.append(index)
            print(index, f" {confidence:.2f} {label}, {'-'*20}")
        else:
            print(index, f" {confidence:.2f} {label}")


    ica_weights_path = f"ica/{subject}_{session}_task-sentences_icaWeights.fif"
    ica = read_ica(ica_weights_path)
    ica.exclude = exclude_components
    raw = ica.apply(raw, n_pca_components=45)

    return raw

    
    
    



if __name__ == '__main__':
    # General Variables
    DATASET_FOLDER_PATH = 'ds004279-download'  # Here all the subjects are stored
    ARTIFACTS_FOLDER_PATH = 'artifacts'  # Artifacts folder for each solver
    SAVE_FOLDER = 'npz'    # Default is current directory
    
    # Step 0. Load aux info
    index2stim = load_stimulus()

    l_freq = 2
    h_freq = 50

    # Find subject and sessions
    subjects_and_sessions = []  # Tuple of subject number and session number
    for sub in os.listdir(DATASET_FOLDER_PATH):
        if 'sub-' in sub:
            for ses  in os.listdir(os.path.join(DATASET_FOLDER_PATH, sub)):
                if 'ses-' in ses:
                    subjects_and_sessions.append((sub, ses)) 
    


    EPOCHS = {}
    # Step 1. Load data
    for subject, session in tqdm(subjects_and_sessions):
        print(subject, session)
        raw = load_subject(subject, session, DATASET_FOLDER_PATH, preload=True)  # triggers_sequencial
        
        # Step 2. Add montage to data
        raw = add_montage(raw)

        # Step 3. Remove bad ica components
        raw = remove_ica_componenets(raw, subject, session, rejection_coinfidence=0.90)
    
        # Step 5. Pre-process data
        raw = pre_processing(raw, l_freq=l_freq, h_freq=h_freq, reference='average', only_language=True)
        
        # Step 6. Create epochs
        # TODO: Darle una mirada y hacer un par de analysis con los epochs para ver que sale
        create_epochs(raw, EPOCHS, subject, session)
        
        
    # Step 7. Save data
    final_file = f"language_average_{l_freq}-{h_freq}hz_icaLabel95confidence_eyes_60sessions.npz"  # Extended: stars one second before the onset of the stimulis
    save_data(EPOCHS, os.path.join(SAVE_FOLDER, final_file))
        

       