#Matthew Bitter - MCS-DS
#CS498
#April 7th, 2018 - HW7 Problem 2

#Import libraries
import pandas as pd
import numpy as np
import imageio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Configuration parameters, number of clusters and iterations
n_clusters = 10
iternations = 100

#loading image and reshaping to one row per pixel (rgb)
img = np.array(imageio.imread('smallsunset.jpg'))
imgf = img.reshape((img.shape[0]*img.shape[1], 3)).astype(np.uint8)

#selecting clusters using kmeans with a random seed.
km = KMeans(n_clusters=n_clusters, random_state=8).fit(imgf)
#assiging cluster centers
cc = np.round(km.cluster_centers_)
#assgning cluster labels
label = km.labels_

#turning into pandas dataframe for convenience of pivot_table function
imgfdf = pd.DataFrame(imgf)
imgfdf['label'] = label #assigning label to the df
#storing a backup array for later
imgdup = np.array(imgfdf)

### Pi initialization
######################
# by counting pixels assigned to cluster from kmeans
#pi sums to one
picount = (imgfdf.pivot_table(index='label',aggfunc=len))
picount = picount/picount.iloc[:,0].sum()
picount = np.array(picount.iloc[:,0])

### W initialization
######################
#setting up empty arrays
xminusu = np.zeros(shape=(imgf.shape[0],3))
wtoplist = np.zeros(shape=(n_clusters,imgf.shape[0]))
pitracker = np.zeros(shape=(iternations,n_clusters))
utracker = np.zeros(shape=(iternations,3))
distancetracker = np.zeros(shape=(n_clusters,imgf.shape[0]))

#calculating the distance "dmin" by looping over clusters and removing the cluster center
#then squaring and summing all the rows up to give distance per pixel. keeping track of them per cluster
for i in range(0,n_clusters):
    xminusu[:, 2] = imgdup[:, 2] - cc[i][2]
    xminusu[:, 1] = imgdup[:, 1] - cc[i][1]
    xminusu[:, 0] = imgdup[:, 0] - cc[i][0]

    distance = np.sqrt(np.sum(np.square(xminusu), axis=1))
    distancetracker[i] = distance

#finding the smallest distance to each pixel and storing the value
distancesmall = np.round(np.amin(distancetracker,axis=0)) #rounding them off since they are pixels


#calculating w by looping over clusters. calculating the x - u for each pixel.
#squaring the different and summing up all the rows to give number of pixels array for each cluster
#then *-1 to make it negative and divide by 2, then subtract the distance to ensure no precision issues
#the calculation in the power has negative numbers before exponent.
#exponent the value and then times by pi. the denominator is just the sum because i am no longer in log space.
for i in range(0,n_clusters):
    xminusu[:, 2] = imgdup[:, 2] - cc[i][2]
    xminusu[:, 1] = imgdup[:, 1] - cc[i][1]
    xminusu[:, 0] = imgdup[:, 0] - cc[i][0]

    wtop = np.exp(((np.sum(np.square(xminusu),axis=1)-np.square(distancesmall))*-1)/2) #numerator
    wtoppi = wtop*picount[i] #numerator
    wtoplist[i] = wtoppi
wbot = wtoplist.sum(axis=0) #denominator
w = wtoplist/wbot

### Start of EM algorithim loops
################################
for i in range(0,iternations):
    #calculating u. matrix multiplication of X and W
    utop = np.dot(imgdup[:, :3].T, w.T) #leave a 3x10
    ubot = w.sum(axis=1) #sum up w across pixels to leave 1x10
    u = utop / ubot #divide to get u pixels
    utracker[i] = u[:,0]

    # calculating pi. same bot that was used in u over number of pixels
    pi = ubot / imgf.shape[0]
    pitracker[i] = pi

    #calculating distance dmin to use in w calculation same logic as in initialization
    #looping over clusters to calculate distance
    for f in range(0, n_clusters):
        xminusu[:, 2] = imgdup[:, 2] - u[2][f]
        xminusu[:, 1] = imgdup[:, 1] - u[1][f]
        xminusu[:, 0] = imgdup[:, 0] - u[0][f]

        distance = np.sqrt(np.sum(np.square(xminusu), axis=1))
        distancetracker[f] = distance

    distancesmall = np.round(np.amin(distancetracker, axis=0))

    #looping over clusters to calculate w. same logic as in initialization
    #x minus u and stores the results for each pixel. square the result and sum up giving number of pixels
    #subtract the distance calculated in the distance loop for each pixel. *-1 to get negative and div by 2
    #exponent the result to get out of log space and multiply by pi. store this for each cluster.
    #the denominator is the sum of the numerator
    for k in range(0, n_clusters):
        xminusu[:, 2] = imgdup[:, 2] - u[2][k] #x - u
        xminusu[:, 1] = imgdup[:, 1] - u[1][k]
        xminusu[:, 0] = imgdup[:, 0] - u[0][k]

        #wtop = np.exp(((np.sum(np.square(xminusu), axis=1)) * -1) / 2)
        wtop = np.exp(((np.sum(np.square(xminusu), axis=1) - np.square(distancesmall)) * -1) / 2)
        wtoppi = wtop * pi[k] #w numerator * pi

        wtoplist[k] = wtoppi #numerator

    wbot = wtoplist.sum(axis=0) #denominator
    w = wtoplist / wbot[:, None].T


#finding the best cluster per pixel
topclusterpixel = np.argmax(w,axis=0)

#generating image using the mean pixels from u
EMimg = u[:,topclusterpixel]
EMimg = EMimg.astype(np.uint8)
EMreshape = EMimg.T.reshape((img.shape[0], img.shape[1], 3))
plt.imshow(EMreshape) #EM image
#plt.imshow(img)  #original image

#convergence plots to indicate when we can stop the iterations
#difference of absolute u's over iterations
udiff = np.abs(utracker[:-1,:] - utracker[1:,:])
plt.plot(udiff[:,0])

#differents of absolute pi's over iterations
pidiff = np.abs(pitracker[:-1,:] - pitracker[1:,:])
plt.plot(pidiff[:,0])
