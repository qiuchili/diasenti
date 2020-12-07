

## Instructions to run the code

### Download the datasets

+ Dialogue Datasets: https://www.dropbox.com/s/kojje169drighei/avec_iemocap_meld.zip?dl=0
  + Containing AVEC, IEMOCAP and MELD datasets
  + Datasets come from https://github.com/SenticNet/conv-emotion
    + iemocap_data.pkl, meld_data_original.pkl and AVEC/* are original datasets in the github
    + iemocap_data_act.pkl and meld_data_act.pkl are the datasets with dialogue act annotations from https://github.com/bothe/EDAs
    + meld_data.pkl is the MELD dataset with dialogue act annotations and pretrained 300d visual features for each utterance

### Do A Single Run (train/valid/test) 

1. Set up the configurations in config/run.ini
2. python run.py -config config/run.ini

#### Configuration setup

+ Dialogue
  + **mode = run**
  + **pickle_dir_path = /path/to/datasets/**. The absolute path of the folder storing the datasets.
  + **wordvec_path = /path/to/glove_embedding/**. The path of the glove embedding file.
  + **dataset_name in `{'iemocap','meld','avec'}`**. Name of the dataset.
  + **features in `{'acoustic','visual','textual'}`**. Multiple modality names should be joined by ','. 
  + **label = emotion**. Only emotion recognition is supported for the time being. 
  + **dialogue_context in `{'True','False'}`**. Whether the data is converted to context + current utterance format (the context of a certain length is extracted for each utterance in the dialogue).
  + **context_len**. Length of the context. Requires **dialogue_context = True**.
  + **seed**. The random seed for the experiment.
  + **load_model_from_dir in `{'True','False'}`**. Whether the model is loaded from a saved file.
  + **dir_name**. The directory storing the model configurations and model parameters. Requires **load_model_from_dir = True**.
  + **fine_tune in `{'True','False'}**. Whether you want to train the model with the data. 
  + **model specific parameters**. For running a model on the dataset, uncomment the respective area of the model and comment the areas for the other models. Please refer to the model implementations in /models/dialogue/ for the meaning of each model specific parameter.
    + supported models include but are not limited to:
      + BC-LSTM
      + CMN
      + ICON
      + DialogueRNN
      + DialogueGCN
      + Multimodal-Transformer
      + EF-LSTM
      + LF-LSTM
      + TFN
      + MFN
      + MARN
      + RMFN
      + LMF
      + Vanilla-LSTHM
      + QMN
    + Each model is only compatible with **dialogue_context = True** or **dialogue_context = False**. For this information, please see the respective block for each model. 
  
### Grid Search for the Best Parameters
1. Set up the configurations in config/grid_search.ini. Tweak a couple of fields in the single run configurations, as instructed below.
2. Write up the hyperparameter pool in config/grid_parameters/.
3. python run.py -config config/grid_search.ini

#### Configuration setup
+ **mode = run_grid_search**
+ **grid_parameters_file**. The name of file storing the parameters to be searched, under the folder /config/grid_parameters. 
  + the format of a file is:
    + [COMMON]
    + var_1 = val_1;val_2;val_3
    + var_2 = val_1;val_2;val_3
+ **search_times**. The number of times the program searches in the pool of parameters.
+ **output_file**.  The file storing the performances for each search in the pool of parameters. By default, it is eval/grid_search_`{dataset_name}`_`{network_type}`.csv






