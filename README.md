# Multi-label ICD-10 classification and application of post hoc XAI models
Code written by Alexander Dolk and Hjalmar Davidsen

For questions, you can reach us at alexander.dolk@hotmail.com or hjalmar.davidsen@gmail.com

TODO: insert link to master's thesis

## Purpose
The purpose of this project has been to fine-tune the classification model SweDeClin-BERT (**REFERENCE**) on swedish gastrointestinal discharge summaries from the [HEALTH BANK](https://dsv.su.se/en/research/research-areas/health/stockholm-epr-corpus-1.146496), and apply the XAI models LIME and SHAP to explain the classifications. 

In our master's thesis, the explanations by [LIME](https://github.com/marcotcr/lime) and [SHAP](https://github.com/slundberg/shap) has then been evaluated by experts from the medical field. The medical experts' evaluations of LIME and SHAP's explanations has then been compared in order to determine potential differences. This study aims to contribute to the [ClinCode research project](https://ehealthresearch.no/en/projects/clincode-computer-assisted-clinical-icd-10-coding-for-improving-efficiency-and-quality-in-healthcare) by evaluating the post hoc model agnostic XAI methods LIME and SHAP on users, to provide insight of what XAI method might be suitable to incorporate in a Computer-Assisted Coding tool for ICD-10 coding, to deliver explainable and interpretable explanations to users.

## How to get a hold of the dataset used
The Swedish gastrointestinal discharge summaries used to fine-tune the SweDeClin-BERT model, and from which discharge summaries has been taken to be evaluated by medical experts, comes from the Stockholm Electroinc Patient Record (EPR) Gastro ICD-10 Corpus (ICD-10 Corpus). The ICD-10 Corpus is a part of EPRs found in the [HEALTH BANK](https://dsv.su.se/en/research/research-areas/health/stockholm-epr-corpus-1.146496). For questions about the HEALTH BANK, please contact Professor [Hercules Dalianis](https://people.dsv.su.se/~hercules/). 

## Documentation of masters_thesis.py
### To rerun the experiment

Observe that the data used for the experiment needs to be in a csv-file. The first row should describe the column-names in the following way: patientnr,anteckning,<columns for all ICD-codes>. The column patientnr, should contain the patient-number in the data. The column anteckning, should contain the discharge summary text. The remaining columns should contain a value of 0 or 1 to denote false or true value for each label (ICD-10 code).

Call the following functions in this order from the main function. You ought run one function at a time:
  
visualise_ICD_code_distribution

create_subset_of_discharge_summaries

create_training_validation_and_test_sets

cross_validation

fine_tune_model

test_model

get_prediction_for_single_discharge_summary


### Comments to functions and classes
#### class GastroDataset
  text
#### class BertForMultilabelSequenceClassification
  text
#### get_dictionary_from_df
  text
#### display_dictionary
  text
#### visualise_ICD_code_distribution
  The full data is read into the variable *data*. 
The variable *full_name* contains all the column-names for all the ICD-codes.
When creating variable *dict_with_values_over_hundred*, only the ICD-codes with samples over 100 are visualized. 
#### create_subset_of_discharge_summaries
  The full data is read into the variable *data*.
When creating the variable *d*, only the ICD-codes that have been visualized using function visualise_ICD_code_distribution are chosen. 
From *data*, only the selected ICD-code columns are picked out along with columns patientnr and anteckning. The selected data is then saved. 
#### create_training_validation_and_test_sets
  The function is used to create training, validation and test sets using the selected data that has been saved from running function create_subset_of_discharge_summaries.
#### cross_validation
The data without test data from the function create_training_validation_and_test_sets is loaded. 
The tokenizer from the not fine-tuned SweDeClin-BERT model is loaded. 
The not fine-tuned SweDeClin-BERT model is loaded. 
Five-fold cross validation is performed using the not fine-tuned SweDeClin-BERT model, with the hyperparameters that will be used for fine-tuning, except for the number of epochs. The number of epochs used for the cross validation is 10. 
The resulting metrics are printed and depending on the metrics, the number of epochs to use for the fine-tuning is decided. 
#### fine_tune_model
  The training data and validation data are loaded.
The tokenizer from the not fine-tuned SweDeClin-BERT model is loaded. 
The not fine-tuned SweDeClin-BERT model is loaded. 
Fine tuning is performed. Hyperparameters are based on previous studies. The number of epochs is decided based on the result of the five-fold cross validation.
The tokenizer and the model is saved.







### Python version and packages used
Python version 3.7.0 has been used for this project and the following packages has been present in the virtual environment used to run the code in masters_thesis.py:

certifi            2021.10.8
charset-normalizer 2.0.12
click              8.1.3
colorama           0.4.4
cuda-python        11.6.1
cycler             0.11.0
Cython             0.29.28
filelock           3.6.0
fonttools          4.33.3
huggingface-hub    0.5.1
idna               3.3
importlib-metadata 4.11.3
joblib             1.1.0
kiwisolver         1.4.2
matplotlib         3.5.2
numpy              1.21.6
packaging          21.3
pandas             1.1.5
Pillow             9.1.0
pip                22.0.4
pyparsing          3.0.8
PyQt5              5.15.6
PyQt5-Qt5          5.15.2
PyQt5-sip          12.10.1
python-dateutil    2.8.2
pytz               2022.1
PyYAML             6.0
regex              2022.4.24
requests           2.27.1
sacremoses         0.0.53
scikit-learn       1.0.2
scipy              1.7.3
setuptools         39.0.1
six                1.16.0
sklearn            0.0
threadpoolctl      3.1.0
tokenizers         0.12.1
torch              1.7.1+cu110
torchaudio         0.7.2
torchvision        0.8.2+cu110
tqdm               4.64.0
transformers       4.18.0
typing_extensions  4.2.0
urllib3            1.26.9
zipp               3.8.0

