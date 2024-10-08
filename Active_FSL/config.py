##### General Parameters #####

MARGIN = 1.0 # xi
WEIGHT = 1.0 # lambda
EPOCH = 100
MILESTONES = [60] # was [160]
# EPOCHL =  60 # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model
MOMENTUM = 0.9
WDECAY = 5e-4

RANDOM_SEED = 103




### Use only one of the following, depending on which dataset is loaded.

##### NCT #####

#NUM_TRAIN    = 10000 # N
#BATCH        = 128   # B #This would be the test_loader batchsize
#ADDENDUM     = 9     # K: each AL cycle selects K samples; was 9
#CYCLES       = 11
#NUM_SHOTS    = 1     # was 10
#NUM_CLASSES  = 9     # train loader batchsize would be smaller= NUM_CLASSES
#NUM_CLUSTERS = 9
#TRIALS       = 1

#LR = 0.3e-3

##### BREAKHIS #####

NUM_TRAIN    = 5873  # N
BATCH        = 128    # B #This would be the test_loader batchsize
ADDENDUM     = 8     # K: each AL cycle selects K samples
CYCLES       = 11    # use 25 to test
NUM_SHOTS    = 1
NUM_CLASSES  = 8
NUM_CLUSTERS = 8
TRIALS       = 1

LR = 1.2e-3

##### LC25000 #####

#NUM_TRAIN    = 18750 # N
#BATCH        = 128 # was 64
#ADDENDUM     = 5
#CYCLES       = 11
#NUM_SHOTS    = 1
#NUM_CLASSES  = 5
#NUM_CLUSTERS = 5
#TRIALS       = 1

#LR = 0.9e-3