# Active-FSL

This code is based on Learning-Loss-for-Active-Learning .
Please set up the environment according to https://github.com/Mephisto405/Learning-Loss-for-Active-Learning .

Download the pickled NCT dataset from google drive then add "nct_pickle" to the "Active_FSL" folder.

Run base_main.py to start active learning.

# Requirements

- Get the pretrained weights, put them in /Pretrain_FSL_Model/
- If any pickling of datasets is required, put the desired split in /Pickle_Dataset/test/ and /Pickle_Dataset/train/
- Get pickled datasets, upload them to the folders indicated in base_main.py (or variations)


# Nico Branch Changes

base_main.py is a stripped down version and has (will have) many functions for testing query strategies