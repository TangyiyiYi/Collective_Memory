import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
np.random.seed(55555)

compute_lambda = lambda g, c, v: (c + (1-c)*(g*((1-g)**(v-1))))/ (g*((1-g)**(v-1)))

start = time.time()
# Initialize parameters:
w = 20 # features
u = 0.04
cw = 0.55
cp = 0.75
criteria = 1.0
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
studyList = np.empty((int(sum(lists[:,0])), w+ncolExtra))
foils = np.empty(studyList.shape)

ncol = studyList.shape[1]
nStudy = studyList.shape[0]
testLists = np.empty((foils.shape[0], 2*ncol))

# Create arrays with all study and test trials:
n1 = 0
nblocks = 4

for n, list in enumerate(lists):
    listLen = int(list[0])
    listType = int(list[1])
    nItems = int(listLen/nblocks)

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
    n2 = n1 + listLen
    studyList[n1:n2, :] = oldItems 
    foils[n1:n2, :] = newItems
    
    # Create test trials
    studyCount = 0
    testCount = 0
    p = 0
    n3 = 0
    idxOld = [0, 1, 2, 5] # [oldItems, oldItems, oldItems, newItems, newItems, oldItems, newItems, newItems]
    idxNew = [3, 4, 6, 7]
    testItems = [oldItems, newItems]
    counters = [studyCount, testCount]
    testList = np.empty((oldItems.shape[0], 2*oldItems.shape[1]))
    for j in range(nblocks):
        n4 = n3 + nItems   
        if p in idxOld:
            testList[n3:n4, :ncol] = testItems[0][counters[0]:counters[0]+nItems]
            counters[0] += nItems
        elif p in idxNew:
            testList[n3:n4, :ncol] = testItems[1][counters[1]:counters[1]+nItems]
            counters[1] += nItems
        p += 1
        # print(f"nItems = {nItems}, n3={n3}, n4={n4}, counters={counters}")   

        if p in idxOld:
            testList[n3:n4, ncol:] = testItems[0][counters[0]:counters[0]+nItems]
            counters[0] += nItems
        elif p in idxNew:
            testList[n3:n4, ncol:] = testItems[1][counters[1]:counters[1]+nItems]
            counters[1] += nItems
        p += 1
        n3 = n4       

    testLists[n1:n2, :] = testList
    n1 = n2

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
correctResp = np.where((testLists[:, ncol-1]==1)&(testLists[:, -1]==0), 0, \
            np.where((testLists[:, ncol-1]==0)&(testLists[:, -1]==1), 1, \
            np.where((testLists[:, ncol-1]==1)&(testLists[:, -1]==1), 11, 22))) #0= left is old, 1= right is old, 11= both are old, 22=both are new
# Test phase
nPure = len(lists[lists[:,1] != 3])
nMix = len(lists[lists[:,1] == 3])
nbars = nPure*4 + nMix*12
accAll = np.empty((nbars, 6))
idxBar = 0
n1 = 0

# Create matrix that has all the needed information
responses = np.full((nStudy, 12), np.nan) #list type, list length, list index, left stimulus type, right stimulus type, left stimulus old/new, right stimulus old/new, correct resp, phi left, phi right, model resp, accuracy 
responses[:, 0]= testLists[:, -4] # list type
responses[:, 1]= testLists[:, -3] # list length
responses[:, 2]= testLists[:, -2] # list index
responses[:, 3]= testLists[:, w+1] # left stimulus type 
responses[:, 4]= testLists[:, -5] # right stimulus type
responses[:, 5]= testLists[:, w+5] # left stimulus old/new
responses[:, 6]= testLists[:, -1] # right stimulus old/new
responses[:, 7] = correctResp

for n, list in enumerate(lists):
    listLen = int(list[0])
    listType = int(list[1])

    currentProbes = testLists[testLists[:, -2] == n, :] # element before last gives the list index
    currentTraces = traces[traces[:, -2] == n, :]

    phis = np.zeros((currentProbes.shape[0], 2))
    for i, trial in enumerate(currentProbes):
        item1 = trial[:ncol]
        item2 = trial[ncol:]
        lambdas = np.zeros((currentTraces.shape[0], 2))

        for j, trace in enumerate(currentTraces):
            lambdaFeature1 = np.where(trace[w+1]==item1[w+1], np.where(trace[:w] == 0, 1, np.where(trace[:w]==item1[:w], compute_lambda(g, item1[w], trace[:w]), (1-item1[w]))), np.nan)
            lambdaFeature2 = np.where(trace[w+1]==item2[w+1], np.where(trace[:w] == 0, 1, np.where(trace[:w]==item2[:w], compute_lambda(g, item2[w], trace[:w]), (1-item2[w]))), np.nan)
            lambdas[j, 0] = np.prod(lambdaFeature1)
            lambdas[j, 1] = np.prod(lambdaFeature2)

        phis[i, :] = np.nanmean(lambdas, axis=0) # a signle phi value for each test item
    responses[n1:n1+listLen, (-4, -3)] = phis # phi for both items
    responses[n1:n1+listLen, -2] = np.argmax(phis, axis=1) # Index of the largest item in each row - #0= left is old, 1= right is old,
    n1 += listLen

responses[:, -1] = np.where(responses[:, -2] == responses[:, 7], 1, 0)

dfResp = pd.DataFrame(responses, columns=["listType", "ntrials", "listIdx", "leftStimType", "rightStimType", "leftStimON", "rightStimON", "corResp", "leftStimPhi", "rightStimPhi", "modelResp", "acc"])
# dfResp["listType"] = dfResp.listType.map({1:"words", 2:"pictures", 3:"mixed"})
dfResp["leftStimType"] = dfResp.leftStimType.map({1:"word", 2:"picture"})
dfResp["rightStimType"] = dfResp.rightStimType.map({1:"word", 2:"picture"})
dfResp["leftStimON"] = dfResp.leftStimON.map({1:"old", 0:"new"})
dfResp["rightStimON"] = dfResp.rightStimON.map({1:"old", 0:"new"})
dfResp["condCoded0"] = dfResp.corResp.map({0:"on", 1:"no", 11:"oo", 22:"nn"})
dfResp["respCoded"] = dfResp.modelResp.map({0:"on", 1:"no", 11:"oo", 22:"nn"})


# Code different conditions for plotting
# dfResp["condCoded"] = "xxx"
# dfResp.loc[(dfResp.listLen == 12)&(dfResp.listType != "mixed"), "condCoded"] = "Pure-12"
# dfResp.loc[(dfResp.listLen == 24)&(dfResp.listType != "mixed"), "condCoded"] = "Pure-24"
# dfResp.loc[(dfResp.listType == "mixed")&(dfResp.leftStimType == dfResp.rightStimType), "condCoded"] = "Mixed"

# dfResp.loc[(dfResp.listType == "mixed")&(dfResp.leftStimType != dfResp.rightStimType)&(dfResp.corResp=="on")&(dfResp.leftStimType == "word"), "condCoded"] = "word is old"
# dfResp.loc[(dfResp.listType == "mixed")&(dfResp.leftStimType != dfResp.rightStimType)&(dfResp.corResp=="no")&(dfResp.rightStimType == "word"), "condCoded"] = "word is old"

# dfResp.loc[(dfResp.listType == "mixed")&(dfResp.leftStimType != dfResp.rightStimType)&(dfResp.corResp=="on")&(dfResp.leftStimType == "picture"), "condCoded"] = "picture is old"
# dfResp.loc[(dfResp.listType == "mixed")&(dfResp.leftStimType != dfResp.rightStimType)&(dfResp.corResp=="no")&(dfResp.rightStimType == "picture"), "condCoded"] = "picture is old"

# dfResp.loc[dfResp.corResp.isin(["on", "no"]), "corResp"] = "no/on"
# dfResp.loc[dfResp.modelResp.isin(["on", "no"]), "modelResp"] = "no/on"

# dfRespAgg = pd.DataFrame(dfResp.groupby(["listType","listLen", "leftStimType", "rightStimType", "condCoded", "corResp", "modelResp"]).agg(
#     avgAcc=("acc","mean"),
#     nTrials = ("listType", "count")))
# dfRespAgg.reset_index(inplace=True)
# # dfRespAgg.drop(dfRespAgg[dfRespAgg.corResp.isin(["nn", "oo"])].index, inplace=True)#.reset_index(inplace=True)

end = time.time()
print(f"Time elapsed in minutes = {(end-start)/60}")
dfResp.to_csv(f"/media/zainab/D/PhD/Research/Forced Choice/Coding/230826_Block2PredData_{cw}_{cp}_{nLists}.csv")
# dfRespAgg.to_csv(f"/media/zainab/D/PhD/Research/Forced Choice/Coding/230509_Block2Pred_{cw}_{cp}.csv") 
# dfRespAgg
# print(f"Average accuracy = {accAll}")    
# dfAcc = pd.DataFrame(accAll, columns=["idx", "length", "type", "acc", "hits", "FA"])
# dfAccAgg = (dfAcc.groupby(["length", "type"]).agg('mean')).reset_index()
# plt.scatter(dfAccAgg.length, dfAccAgg.acc, label="acc")
# plt.scatter(dfAccAgg.length, dfAccAgg.hits, label="hits")
# plt.ylim(0, 1)
# plt.xlabel("list length")
# plt.ylim("avg acc")
# plt.legend()
# plt.show()