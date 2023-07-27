# Large Spanish EEG

EEG: silent and perceive speech on 30 spanish sentences 

Large Spanish Speech EEG [dataset](https://openneuro.org/datasets/ds004279)
Authors
<ul>
  <li>Carlos Valle</li>
  <li>Carolina Méndez-Orellana</li>
  <li>María Rodríguez-Fernández</li>
</ul>



The dataset is part of a publication by the authors: Carlos Valle, Carolina Méndez-Orellana, María Rodríguez-Fernández and Christian Herff

---
### **Subject-independent decoding of perceived sentences from EEG signals using artificial neural networks**

Abstract:
Decoding speech from brain activity can enable communication for individuals with speech disorders. Deep neural networks have shown great potential for speech decoding applications, but the large data sets required for these models are usually not available for neural recordings of speech impaired subjects. Harnessing data from other participants would thus be ideal to create speech neuroprostheses without the need of patient-specific training data.
In this study, we recorded 60 sessions from 56 healthy participants using 64 EEG channels and developed a neural network capable of subject-independent classification of perceived sentences. We found that sentence identity can be decoded from subjects without prior training achieving higher accuracy than mixed-subject models.
The development of subject-independent models eliminates the need to collect data from a target subject, reducing time and data collection costs during deployment. These results open new avenues for creating speech neuroprostheses when subjects cannot provide training data.  

**ARTICLE UNDER REVIEW**

Cite:
```
@article{Valle,
    doi       = {},
    url       = {},
    year      = {},
    publisher = {},
    volume    = {},
    number    = {},
    pages     = {2826},
    author    = {Carlos Valle, Carolina Méndez-Orellana, María Rodríguez-Fernández and Christian Herff},
    title     = {Subject-independent decoding of perceived sentences from EEG signals using artificial neural networks},
    journal   = {}
}
``````



# Repository content
To download the dataset you can use the tools provided by openNeuro


**Download from AWS S3**
```
aws s3 sync --no-sign-request s3://openneuro.org/ds004279 ds004279-download/
```

**Download using Node.js with @openneuro/cli**    <- **Recommended for reviewers**
```
openneuro download --snapshot 1.1.0 ds004279 ds004279-download/
```

**Download with DataLad**
```
datalad install https://openneuro.org/git/1/ds004279
```


To recreate the experiments, you first need to run `main_pipeline.py` followed by any of the `three model_*.ipynb`.
## Files
* `main_pipeline.py` : Contains all the details in terms of signal processing, epochs creating, artifact removal and ica decomposition. This script generates an `.npy` file that can be read in the notebooks.
* `model_*.ipynb`: Contains the experiments for condition classification (rest, perception and production), pairs of sentences and window size effect in the decoding accuracy.

* `get_ica.py`: This script allows to re-compute the ica-weights matrix included in ica_label folder, as well as the automatic classification of the components by [ica label](https://github.com/mne-tools/mne-icalabel) mne implementation.

* `stimulus_dict.json`: Contains the transcription of the sentences used (Spanish).

## Folders
The folders included in this repository have auxiliary information that can be of interest to others.  
* `ica`: Contains pre-computed ICA weights matrix using 45 components in `.fif` format. In addition, `.json` files have the classification of the components performed by ica label mne 
* `sentences`: Original sentences presented to the volunteers in `.wav`.
* `praat_annotations`:  Contains `.TextGrid` files with word timestamps of each sentence. 

## Contact
Please contact us at this e-mail address if you have any question: cgvalle@uc.cl.


