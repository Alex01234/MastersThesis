# Multi-label ICD-10 classification and application of post hoc XAI models
Code written by Alexander Dolk and Hjalmar Davidsen

For questions, you can reach us at alexander.dolk@hotmail.com or hjalmar.davidsen@gmail.com

TODO: insert link to master's thesis

## Purpose
The purpose of this project has been to fine-tune the classification model SweDeClin-BERT (**REFERENCE**) on swedish gastrointestinal discharge summaries from the [HEALTH BANK](https://dsv.su.se/en/research/research-areas/health/stockholm-epr-corpus-1.146496), and apply the XAI models LIME and SHAP to explain the classifications. 

In our master's thesis, the explanations by [LIME](https://github.com/marcotcr/lime) and [SHAP](https://github.com/slundberg/shap) has then been evaluated by experts from the medical field. The medical experts' evaluations of LIME and SHAP's explanations has then been compared in order to determine potential differences. This study aims to contribute to the [ClinCode research project](https://ehealthresearch.no/en/projects/clincode-computer-assisted-clinical-icd-10-coding-for-improving-efficiency-and-quality-in-healthcare) by evaluating the post hoc model agnostic XAI methods LIME and SHAP on users, to provide insight of what XAI method might be suitable to incorporate in a Computer-Assisted Coding tool for ICD-10 coding, to deliver explainable and interpretable explanations to users.

## How to get a hold of the dataset used
The Swedish gastrointestinal discharge summaries used to fine-tune the SweDeClin-BERT model, and from which discharge summaries has been taken to be evaluated by medical experts, comes from the Stockholm Electroinc Patient Record (EPR) Gastro ICD-10 Corpus (ICD-10 Corpus). The ICD-10 Corpus is a part of EPRs found in the [HEALTH BANK](https://dsv.su.se/en/research/research-areas/health/stockholm-epr-corpus-1.146496). For questions about the HEALTH BANK, please contact Professor [Hercules Dalianis](https://people.dsv.su.se/~hercules/). 

## Documentation
>text<
