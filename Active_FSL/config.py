''' Configuration File.
'''

##### NCT #####

#NUM_TRAIN = 10000 # N
#BATCH     = 128 # B #This would be the test_loader batchsize
#ADDENDUM  = 9 # K: each AL cycle selects K samples; was 9
#CYCLES = 15
#NUM_SHOTS=1 # was 10
#NUM_CLASSES= 9  #train loader batchsize would be smaller= NUM_CLASSES
#NUM_CLUSTERS=9
#TRIALS = 1

##### BREAKHIS #####

#NUM_TRAIN = 1400 # N
#BATCH     = 64 # B #This would be the test_loader batchsize
#ADDENDUM  = 8 # K: each AL cycle selects K samples
#CYCLES = 16 # use 25 to test
#NUM_SHOTS=1
#NUM_CLASSES= 8
#NUM_CLUSTERS=8
#TRIALS = 1

##### LC25000 #####

NUM_TRAIN    = 4000 # N
BATCH        = 64
ADDENDUM     = 5
CYCLES       = 16
NUM_SHOTS    = 1
NUM_CLASSES  = 5
NUM_CLUSTERS = 5
TRIALS       = 1

##### General Parameters #####

MARGIN = 1.0 # xi
WEIGHT = 1.0 # lambda
EPOCH = 200
LR = 1e-3 # best success with: NCT 1e-4, BreakHis 1e-3, LC25000 1e-3
MILESTONES = [160]
EPOCHL = 120 # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model
MOMENTUM = 0.9
WDECAY = 5e-4

RANDOM_SEED = 109