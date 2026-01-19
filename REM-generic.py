import numpy as np
import matplotlib.pyplot as plt
np.random.seed(55555)

# Compute likelihood ratio based on equation 4A in the REM paper
compute_lambda = lambda g, c, v: (c + (1-c)*(g*((1-g)**(v-1))))/ (g*((1-g)**(v-1)))

# Initialize parameters:
w = 20 # number of features
u = 0.04 # parameter u: probability of storing a value for a feature
c = 0.7 # parameter c: probability of correctly copying a feature
criteria = 1.0
nSteps = 10 # number of storage steps
gH = 0.42 # gH parameter in the geometric distribution
g = 0.4 # g parameter in the geometric distribution

nStudy = 100 # number of study trials
nTest = nStudy * 2 # number of test trials

# Generate features for study and test items based on the geometric distribution
studyList = np.column_stack((np.random.geometric(gH, (nStudy, w)), np.ones((nStudy)))) # Study words - label = 1
foils = np.column_stack((np.random.geometric(g, (nStudy, w)), np.zeros((nStudy)))) # distractors - label:0

testList = np.vstack((studyList, foils)) # Last column indicates whether it is a study or a test word
np.random.shuffle(testList)

# Initialize arrays to store study items
traces = np.zeros((studyList.shape[0], studyList.shape[1]-1))
lambdas = np.zeros((studyList.shape[0]))
correctResp = np.zeros((nTest))
phi = np.zeros((nTest))

# Store study items
for i in range(nSteps):
    for k, item in enumerate(studyList):
        store = np.random.choice([0, 1], w, p=[1-u, u]) # Which features to be stored at this time point?
        copyCorr = np.random.choice([0, 1], w, p=[1-c, c]) # Will the stored features be copied correctly?

        trace = traces[k, :]
        traces[k, :] = np.where((store>0)&(copyCorr>0)&(trace==0), item[:w], np.where((store>0)&(copyCorr==0)&(trace==0), np.random.geometric(g), trace)) 

# Test phase
for k, probe in enumerate(testList):
    for j, trace in enumerate(traces):
        lambdaFeature = np.where(trace == 0, 1, np.where(trace==probe[:w], compute_lambda(g, c, trace), (1-c)))
        lambdas[j] = np.prod(lambdaFeature)

    correctResp[k] = probe[w] # the correct response that the model should predict
    phi[k] = np.mean(lambdas)


response = np.where(phi<criteria, 0, 1) # 0=new, and 1=old
acc = np.where(response==correctResp, 1, 0)

print(f"Average accuracy = {np.mean(acc)}")    
        

