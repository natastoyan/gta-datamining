# -*- coding: cp866 -*-

import sys
import numpy as np
import scipy as sp
from pandas import *
from matplotlib import pyplot as plt
from math import log
from datetime import timedelta
from scipy.cluster.vq import kmeans, vq, whiten
from pybrain.tools.shortcuts import buildNetwork 
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.structure.modules import SoftmaxLayer, SigmoidLayer, LinearLayer, TanhLayer 
from pybrain.tools.xml.networkwriter import NetworkWriter


# с авариями: c:\gta-data\exportGrnR3e2.csv, c:\gta-data\exportMzgR8e1.dsv, c:\gta-data\exportAlmR3e1.csv, c:\gta-data\exportNkgR9e3.dsv, 
# exportMzgR8e3.dsv, exportPrkR1e11.dsv, #exportVtkR1e2.dsv, exportNkgR9e2.dsv (delimiter = ',''), 
# exportNkgR10e1.dsv, exportGrnR6e1.dsv, c:\data\exportChaR6e2.dsv, exportIgrR1e1.dsv
# без аварий: c:\gta-data\exportGrnR2e4.dsv



PARAMETERS = [
    'SD','PINB','TINB',
    'Q','POUTB','TOUTB',
    'REVHP','REVLP','REVWE',
    'TFIRE','CF'
]
EVENTS = [
    'AVARIA', 'REMONT', 'RABOTA',
    'REZERV', 'NOINFO'
]

def read_dict(path):
    dic={}
    for line in open(path):
        key,value=line.strip().split(':')
        dic[key]=float(value)  #dic[key]=value
    return dic


def loaddata(path):
    df = read_csv(
        path,';', index_col=["TIMESTAMP"],
        parse_dates=["TIMESTAMP"], dayfirst=True
        )
    return df


def eventshandling(df):
    for event in EVENTS:
        df[event] = 0 
    for i in range(len(df)):
        if df['EVENT'][i] == '\xc0\xe2\xe0\xf0\xe8\xff':#avaria
            df['AVARIA'][i] = 1
        if df['EVENT'][i] == '\xc2 \xf0\xe5\xec\xee\xed\xf2\xe5':#v remonte
            df['REMONT'][i] = 1
        if df['EVENT'][i] == '\xc2 \xf0\xe0\xe1\xee\xf2\xe5':#v rabote
            df['RABOTA'][i] = 1
        if df['EVENT'][i] == '\xc2 \xf0\xe5\xe7\xe5\xf0\xe2\xe5':#v rezerve
            df['REZERV'][i] = 1
        if df['EVENT'][i] == '\xcd\xe5\xf2 \xe4\xe0\xed\xed\xfb\xf5':#net dannyh
            df['NOINFO'][i] = 1
    del df['EVENT']
    return df


def norm(arr):
    arr = (arr - np.mean(arr))/(np.max(arr) - np.min(arr))
    return arr


def normalize(data_frame, param, mean, minmax):
    data_frame[param] = (
        (data_frame[param] - mean[''.join(['AVG(',param,')'])]) / (
            minmax[''.join(['MAX-MIN(',param,')'])]
        )
    )
    return data_frame


def normalize_data(df):
    # Read dictionary with min/max values for normalization.
    minmax_dict = read_dict(r'c:\gta-data\minmax-sub.txt')
    mean_dict = read_dict(r'c:\gta-data\mean.txt')
    for param in PARAMETERS:
        df = normalize(df, param, mean_dict, minmax_dict)
        for i in range(1, 4):# Some values are too small, they should be multiplied by 10 or 100.
            if abs(df[param]).max() < 0.00009 * (10**i):
                df[param] = df[param] * 10**(4-i)
    return df


def segmentation(df):
    dfs = df.resample('90Min')
    dfs = dfs.dropna(subset=PARAMETERS, how='all')
    dfs = dfs.ffill()
    for i in range(len(dfs)):
        if dfs.AVARIA[i] <> 0:
            dfs.AVARIA[i-32:i+1] = 0.9
    for i in range(len(dfs)):
        if dfs.RABOTA[i] <> 0:
            print sum(dfs.REZERV[i-16:i])
            if sum(dfs.REZERV[i-16:i]) == 0 and sum(
             dfs.REMONT[i-16:i]) == 0 and sum(
             dfs.AVARIA[i-16:i])== 0:
                dfs.RABOTA[i-16:i+1] = 0.9
    return dfs


def clustering(df):
    dfc = df
    dfc = whiten(dfc) # sort of normalization
    data = np.ndarray((len(dfc), len(dfc.axes)), buffer = dfc[PARAMETERS].values)
    centroids, dis = kmeans(data, 3)
    centroids = whiten(centroids)
    idx, _ = vq(data, centroids)
    badcentroids = 0
    if (len(centroids) <= 2 or np.isnan(centroids).any()
            or centroids.any() == 0.0):
        badcentroids += 1
        clustering(df)
    else: 
        print 'number of try:', badcentroids
        print 'distortion:', dis
        plt.plot(data[idx==0,0],data[idx==0,1],'ob', 
                 data[idx==1,0],data[idx==1,1],'or',
                 data[idx==2,0],data[idx==2,1],'oy')
        plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
        plt.show()
        print 'Are clusters ok? y/n'
        if raw_input() == 'n':
            try:
                clustering(df)
            except: pass
        else: pass
    return centroids, data, idx  


def plot_dataframe(df):
    for param in PARAMETERS:
        plt.plot(df.index, df[param], label = param)
    # for event in (EVENTS):
        # plt.plot(df.index, df[event],  label = event)
    try:
        # plt.plot(df.index, df.AVARIA[df.AVARIA <> 0], 'ro')
        # plt.plot(df.index, df.RABOTA[df.RABOTA <> 0], 'bo')
        for i in range(len(df)):
            if df.AVARIA[i] <> 0:
                plt.plot(df.index[i], df.AVARIA[i], 'ro')
        for i in range(len(df)):
            if df.RABOTA[i] <> 0:
                plt.plot(df.index[i], df.RABOTA[i], 'bo')
        
    except: 
        print 'Error while plotting events'
    try:
        plt.plot(df.index, df.CLUSTER, 'r--', label = 'CLUSTER')
    except: 
        print 'Error while plotting cluster'
    plt.legend(bbox_to_anchor=(1.05, 1), loc=9, borderaxespad=0.)
    plt.show()


def plot_cluster(data, centroids, idx):
    plt.plot(data[idx==0,0],data[idx==0,1],'ob',
             data[idx==1,0],data[idx==1,1],'or')
    plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
    plt.show()


def map_dataframe_and_clusters(df, idx):
    #df['CLUSTER'] = norm(idx)
    df['CLUSTER'] = idx
    return df

def main():
    df = loaddata(sys.argv[1])
    #plot_dataframe(df)
    df = eventshandling(df)
    #plot_dataframe(df)    
    # print 'index, REZERV, REMONT, RABOTA'
    # for i in range(len(df)):
        # if df.REZERV[i] <> 0 or df.REMONT[i] <> 0 or df.RABOTA[i] <> 0:
            # print df.index[i], df.REZERV[i], df.REMONT[i], df.RABOTA[i]
    dfn = normalize_data(df)
    #plot_dataframe(dfn)
    dfs = segmentation(dfn)
    #plot_dataframe(dfs)
    # for i in range(len(dfs)):
        # if dfs.RABOTA[i] <> 0:
            # print dfs.index[i]
    centroids, data, idx = clustering(dfs)
    dfc = map_dataframe_and_clusters(dfs, idx)
    plot_dataframe(dfc)
    print dfc

    #plot_cluster(centroids, data, idx) #doesn't work


if __name__ == '__main__':
    main()


#NEURAL NETWORK
#Teaching and testing network
# dfu = read_csv('c:/data/segments.csv',';', index_col=["TIMESTAMP"], parse_dates=["TIMESTAMP"], dayfirst=True)
# ind = [i for i in range(len(dfu)) if dfu['SEGMENT'][i]>0.9]
# labels = list()

# for i in range(len(ind)):
 # if dfu['SEGMENT'][ind[i]]>1: 
  # j = 1 
 # else: 
  # j = 0
 # labels.append(j)

# net = buildNetwork(3, 3, 1)
# net.activate([nan, -1, 1])

# ds = ClassificationDataSet(600, 1, nb_classes=2, class_labels=['bad', 'good']) 
#add to data set segments without empty values
# for par in PARAMETERS:
    # for i in range(len(ind)):
        # j = ind[i]
        # flag = 1
    # while j in range(ind[i], ind[i]+600):
        # if isnan(dfu.CF[j]):
            # flag = 0
        # j+=1
    # if flag:
        # ds.addSample(dfu.CF[ind[i]:(ind[i]+600)], (labels[i], ))


# tries = 10
# bias = True
# fast = False
# previous_error = 100
# epochs = 60
# layer_dim = 0.1
# for _ in xrange(tries):
 # train_ds, test_ds = ds.splitWithProportion(0.7)
 # try_net = buildNetwork(train_ds.indim, int(train_ds.indim*layer_dim), train_ds.outdim, hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=bias, fast=fast)
 # trainer = BackpropTrainer(try_net, train_ds)
 # trainer.trainEpochs(epochs)
 # trnresult = percentError(trainer.testOnClassData(), train_ds['class'])
 # tstresult = percentError(trainer.testOnClassData(dataset=test_ds ), test_ds['class'])
 # print test_ds['target']
 # print "epoch: %4d" % trainer.totalepochs, \
  # " train error: %5.2f%%" % trnresult, \
  # " test error: %5.2f%%" % tstresult
 # if tstresult < previous_error:
            # net = try_net
            # previous_error = tstresult

 # NetworkWriter.writeToFile(net, 'net.xml')
 # layer_dim = layer_dim * 2

# train_ds, test_ds = ds.splitWithProportion(0.7)
# try_net = buildNetwork(train_ds.indim, train_ds.indim*2, train_ds.outdim, hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=bias,
 # fast=fast)
# trainer = BackpropTrainer(try_net, train_ds)
# trainer.trainEpochs(epochs)
# trnresult = percentError(trainer.testOnClassData(), train_ds['class'])
# tstresult = percentError(trainer.testOnClassData(dataset=test_ds ), test_ds['class'])
# print test_ds['target']
# print "epoch: %4d" % trainer.totalepochs, \
 # " train error: %5.2f%%" % trnresult, \
 # " test error: %5.2f%%" % tstresult
 