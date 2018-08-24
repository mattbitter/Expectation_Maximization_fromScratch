#Matthew Bitter
#CS498 - MCS-DS - April 7th, 2018
#HW7 Problem 1

#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load data
pd_doc = pd.read_table("docword.nips.txt",header=None,sep=' ')
pd_voc = pd.read_table("vocab.nips2.txt",header=None,sep=' ')
docword = np.array(pd_doc)
#Transform into rows as docs and columns as words
docwordpd = pd_doc.pivot(index=0, columns=1, values=2)
docwordpd = docwordpd.fillna(0)

#define log sum exp function - very important
def logsumexpc(X):
    x_max = X.max(1)
    #return  np.log(np.exp(X + x_max[:, None]).sum(1)) - x_max
    return x_max + np.log(np.exp(X - x_max[:, None]).sum(1))

#Define pi (not using kmeans) to all be even probability over 30 clusters
pi = np.full((1,30),1/30)


########### P in  Initialization
################################
#Defining P (not using kmeans). probability of word in a cluster so each cluster sums to 1.
#assigning clusters randomly
docwordpd['label'] = np.random.randint(0, 30, docwordpd.shape[0])
#summing word counts for whole cluster
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.pivot_table.html
topicc = np.array(docwordpd.pivot_table(index=['label'],aggfunc=np.sum))
totaltc = topicc.sum(axis=1)
#getting probability
p = topicc/totaltc[:,None]
#adding smoothing to the zero P's and re normalizing again.
p[p == 0] = 0.0000001
p_scaled = p/p.sum(axis=1)[:,None]

######## W in Initialization - probability that document i is assigned to topic/cluster j
############################
#do the w calculation in log space.
#take log of p
p_scaledlog = np.log(p_scaled)
#take log of pi
log_pi = np.log(pi)
docwordpd.drop(labels='label',inplace=True,axis=1)

#matrix multiply X and log(p) ... X * log(p) because when you log it bring the power down
docwordpd.reset_index(drop=True,inplace=True)
inner = np.matmul(docwordpd,p_scaledlog.T)
#add each pi cluster to the product
wtop = (inner + log_pi)

#take logsumexp to get the denominator. very important to avoid precision errors
#subtract the denominator because we are in log space
logw = wtop-logsumexpc(wtop)[:, None]
#take it out of log space and each doc sums to 1
w = np.exp(logw)


######### Start of EM loops
###########################
#variable to track pi's in each loop
pietracker = np.zeros(shape=(20, 30))

#looping through each iteration
for i in range(0,20):
    ### M -Step
    #matrix multiply W and X to give Pn+1
    pplus1top = np.dot(w.T,docwordpd)
    pplus1 = pplus1top / pplus1top.sum(axis=1)[:,None] #getting probability
    pplus1[pplus1 == 0] = 0.0000001 #smoothing
    pplus1prob = pplus1 / pplus1.sum(axis=1)[:,None] #rescaling after smoothing

    #calculate pi n+1 sum of w by number of documents
    piplus1 = np.sum(w, axis=0) / 1500
    pietracker[i] = piplus1
    #np.sum(piplus1)

    ### E Step - calculating w
    #log p
    ppluslog1 = np.log(pplus1prob)
    #log pi
    log_pi1 = np.log(piplus1)
    # matrix multiply X and log(p) - same as doing xi and pi and summing them.
    inner = np.dot(docwordpd,ppluslog1.T)
    #add log pi because we in log space
    wtop = (inner + log_pi1)
    # take logsumexp to get the denominator. very important to avoid precision errors
    logw = wtop-logsumexpc(wtop)[:, None]
    #exponent to take out of log space
    w = np.exp(logw)


#printing pi tracker over iterations and probabilities of final pi
plt.bar(np.arange(len(piplus1)),piplus1)
plt.plot(pietracker)
plt.plot(pietracker[:,4][1:] - pietracker[:,4][:-1])


####Finding top words. empty words were removed from data file directly
sra2 = np.argsort(pplus1prob)
sra1 = sra2[:,:-11:-1]

topwords = pd.DataFrame()
for i in range(0,30):
    topten = pd_voc.iloc[sra1[i],:]
    topten.reset_index(drop=True, inplace=True)
    topten = topten.T
    topten['cluster'] = i
    topwords = topwords.append(topten)

print(topwords)

