# Overview

This is the code repository for the publication "MyriadAL: Active Few Shot Learning for Histopathology", accepted to IEEE CAI 2024. The paper, details of the methodology, and results are available at https://arxiv.org/abs/2310.16161

# Usage

First, install the requirements listed in the Python files. Then, set up directories `nct_pickle/`, `lc25000_pickle/`, and `breakhis_pickle/`, depending on the dataset of interest. Download the desired dataset from its respective online source, and pickle the dataset using the utilities in `Pickle_Dataset/`.

Then, pretrain the model using MoCo with the utilities in `Pretrain_FSL_Model/`, and generate the starting pseudo labels with `Generate_Pseudo_Labels/`.

Replace the paths in `Active_FSL/base_main_moco_model.py` and similar files, depending on the dataset of interest. Also modify the file `config.py` to suit your needs. Use the pretrained model and the generated pseudo labels in the appropriate spots.

The training process can then be started by running `Active_FSL/base_main_moco_model.py`. The results are stored in the log files located in the same folder, or outputted in the command line.
