import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
np.random.seed(55555)
np.set_printoptions(threshold=sys.maxsize)

compute_lambda = lambda g, c, v: (c + (1-c)*(g*((1-g)**(v-1))))/ (g*((1-g)**(v-1)))
cStoreDiff = True # Flag to decide if c used during storage is different from c used for odds computations or not

# Initialize parameters:
w = 20 # features
u = 0.04
# cRatio = 0.75 / 0.55
cp = 0.75 # 0.8 # np.linspace(0.1, 1, 20)
cw = 0.5 # 0.525 # np.linspace(0.1, 1, 20)
cInter = 0.625
criteria = 1
nSteps = 10
gH = g = 0.4

nLists = 5000
listLengths = [12, 24]
listTypes = [1, 2, 3] #["w", "p", "wp"]
listProbs = [1/4, 1/4, 1/2]

lists=np.empty((nLists, 2))
lists[:, 1] = np.random.choice(listTypes, nLists, p=listProbs)
randLeng = np.random.choice(listLengths, nLists)
lists[:, 0] = np.where(lists[:,1]==3, 24, randLeng)

ncolExtra = 6 # Extra info saved for each stimulus other than features
nblocks = 4

studyList = np.empty((int(sum(lists[:,0])), w+ncolExtra))
foils = np.empty(studyList.shape)
ncol = studyList.shape[1]
nStudy = studyList.shape[0]
testLists = np.empty((2*foils.shape[0], ncol))

# Create arrays with all study and test trials:
n1 = 0
n2 = 0

start = time.time()
for n, list in enumerate(lists):
    listLen = int(list[0])
    listType = int(list[1])

    if listType == 1:
        stimType = np.repeat(1, listLen)
        cs = np.repeat(cw, listLen)
    elif listType == 2:
        stimType = np.repeat(2, listLen)
        cs = np.repeat(cp, listLen)
    else:
        stimType = np.hstack((np.tile([1,2], 3), np.repeat([2,1], 3), np.repeat([2,1], 3), np.tile([1,2], 3)))
        cs = np.where(stimType==1, cw, cp)

    # Last column= whether it is a study or a test item, column before last= list index, the one before= list length, the one before= list type, the one before= stimulus type, the one before= c
    oldItems = np.column_stack((np.random.geometric(gH, (listLen, w)),cs, stimType, np.repeat(listType, listLen), np.repeat(listLen, listLen), np.repeat(n, listLen), np.ones((listLen)))) # Study words - label=1
    newItems = np.column_stack((np.random.geometric(g, (listLen, w)),cs, stimType, np.repeat(listType, listLen), np.repeat(listLen, listLen), np.repeat(n, listLen), np.zeros((listLen)))) # distractors - label=0
    
    # Save old and new items
    studyList[n1:n1+listLen, :] = oldItems 
    foils[n1:n1+listLen, :] = newItems
    
    # Create test trials
    testLists[n2:n2+listLen, :] = oldItems
    testLists[n2+listLen:n2+2*listLen, :] = newItems
    np.random.shuffle(testLists[n2:n2+2*listLen, :])

    n1 += listLen
    n2 += 2*listLen

# Store study items
traces = np.zeros((nStudy, ncol))
traces[:, w:] = studyList[:, w:] # list type and other info
for i in range(nSteps):
    store = np.random.choice([0, 1], (nStudy, w), p=[1-u, u]) # Which features to be stored at this time point?    
    for k, item in enumerate(studyList):
        copyCorr = np.random.choice([0, 1], w, p=[1-item[-6], item[-6]]) # Will the stored features be copied correctly? - This is slow, try to optimize it
        trace = traces[k, :w]
        traces[k, :w] = np.where((store[k, :]>0)&(copyCorr>0)&(trace==0), item[:w], np.where((store[k, :]>0)&(copyCorr==0)&(trace==0), np.random.geometric(g), trace)) 

# Code correct responses in each trial
correctResp = np.where((testLists[:, -1]==1), 1, 0) #0= new item, 1= old item

# Create matrix that has all the needed information
responses = np.full((2*nStudy, 9), np.nan) #list type, list length, list index, stimulus type, stimulus old/new, correct resp, phi, model resp, accuracy 
responses[:, 0]= testLists[:, -4] # list type
responses[:, 1]= testLists[:, -3] # list length
responses[:, 2]= testLists[:, -2] # list index
responses[:, 3]= testLists[:, w+1] # stimulus type 
responses[:, 4]= testLists[:, -1] # stimulus old/new
responses[:, 5] = correctResp

# Test phase
n3 = 0
for n, list in enumerate(lists):
    listLen = int(list[0])
    listType = int(list[1])

    currentProbes = testLists[testLists[:, -2] == n, :] # element before last gives the list index
    currentTraces = traces[traces[:, -2] == n, :]

    phis = np.zeros((currentProbes.shape[0]))
    for i, trial in enumerate(currentProbes):
        item1 = trial[:ncol]
        lambdas = np.zeros((currentTraces.shape[0]))

        for j, trace in enumerate(currentTraces):

            if cStoreDiff:
                lambdaFeature1 = np.where(trace[w+1]==item1[w+1], np.where(trace[:w] == 0, 1, np.where(trace[:w]==item1[:w], compute_lambda(g, cInter, trace[:w]), (1-cInter))), np.nan)
            else:
                lambdaFeature1 = np.where(trace[w+1]==item1[w+1], np.where(trace[:w] == 0, 1, np.where(trace[:w]==item1[:w], compute_lambda(g, item1[w], trace[:w]), (1-item1[w]))), np.nan)
            
            lambdas[j] = np.prod(lambdaFeature1)

        phis[i] = np.nanmean(lambdas) # a signle phi value for each test item
    responses[n3:n3+2*listLen, -3] = phis # phi
    responses[n3:n3+2*listLen, -2] = np.where(phis>criteria, 1, 0)
    n3 += 2*listLen

responses[:, -1] = np.where(responses[:, -2] == responses[:, 5], 1, 0)

dfResp = pd.DataFrame(responses, columns=["listType", "ntrials", "listIdx", "stimType", "stimON", "corResp", "stimPhi", "modelResp", "acc"])
# dfResp["listType"] = dfResp.listType.map({1:"words", 2:"pictures", 3:"mixed"})
dfResp["stimType"] = dfResp.stimType.map({1:"word", 2:"picture"})
dfResp["stimON"] = dfResp.stimON.map({1:"old", 0:"new"})
dfResp["condCoded0"] = dfResp.corResp.map({0:"n", 1:"o"})
dfResp["respCoded"] = dfResp.modelResp.map({0:"n", 1:"o"})

if cStoreDiff:
    dfResp.to_csv(f"/media/zainab/D/PhD/Research/Forced Choice/Coding/Modeling/modelPredictions/singleItemRec/oldness-singleItemRec_{criteria}_{cw}_{cp}_{nLists}_cOdds{cInter}.csv")
else:
    dfResp.to_csv(f"/media/zainab/D/PhD/Research/Forced Choice/Coding/Modeling/modelPredictions/singleItemRec/oldness-singleItemRec_{criteria}_{cw}_{cp}_{nLists}.csv")

end = time.time()
print(f"Time elapsed in minutes = {(end-start)/60}")
print(f"Finished the simulation for cp={cp} & cw={cw}")
print(f"average accuracy for this simulation = {np.mean(dfResp.acc)}")

dfResp[["stimON", "acc"]].value_counts()

