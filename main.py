# -*- coding: cp866 -*-

import sys
import numpy as np
import scipy as sp
from pandas import *
from matplotlib import pyplot as plt
from math import *
from pybrain.tools.shortcuts import buildNetwork 
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.structure.modules import SoftmaxLayer, SigmoidLayer, LinearLayer, TanhLayer 
from pybrain.tools.xml.networkwriter import NetworkWriter


# с авариями: c:\gta-data\exportGrnR3e2.csv, exportMzgR8e1.dsv, exportAlmR3e1.csv, exportNkgR9e3.dsv, 
# exportMzgR8e3.dsv, exportPrkR1e11.dsv, #exportVtkR1e2.dsv, exportNkgR9e2.dsv (delimiter = ',''), 
# exportNkgR10e1.dsv, exportGrnR6e1.dsv, c:\data\exportChaR6e2.dsv, exportIgrR1e1.dsv
# без аварий: c:\gta-data\exportGrnR2e4.dsv


DAY = 86400

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
        df[event] = float('NaN') 
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

def plot_data(df):
    for param in PARAMETERS:
        plt.plot(df.index, df[param], label = param)
    for event in (EVENTS):
        plt.plot(df.index, df[event],  label = event)
    #plt.plot(df.index, df['AVARIA'], 'bo', label='AVARIA')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=9, borderaxespad=0.)
    plt.show()

# Segmentation
def segmentation(df):
    df = df.asfreq('1S')
    #где-то здесь должно определяться, как высчитывается параметр за 1,5 часа. сейчас вроде среднее.
    df = df.resample('90Min')
    df = df.dropna(subset=PARAMETERS, how='all')
    df['DATETIME'] = df.index
    df['DELTA'] = (df['DATETIME'] - df['DATETIME'].shift()).fillna(0)
    df['WORK'] = 0
    df['CRASH'] = 0
    df_avaria = df[df.AVARIA.notnull()]
    df_rabota = df[df.RABOTA.notnull()]
    df_rezerv = df[df.REZERV.notnull()]
    df_remont = df[df.REMONT.notnull()]
    # for i in range(len(df)):
        # for j in range(len(df)):
            # if (df.index[i] - 
    df.AVARIA.notnull()


def main():
    print 'Hello, data mining!'
    df = loaddata(sys.argv[1])
    eventshandling(df)
    dfn = normalize_data(df)
    plot_data(dfn)
    dfs = segmentation(df)


if __name__ == '__main__':
    main()


#MARK SEGMENTS WITH ACCEDENTS

# segments = list()
# for i in range(len(df)):
    # if df['SEGMENT'][i] == 1:
        # segments.append(i)
# for i in range(len(df)):
    # if df['LABEL'][i] > 0.9:
        # for j in range(len(segments)):
    # if i > segments[j]:
        # try:
            # if i < segments[j+1]:
                # df['SEGMENT'][segments[j]] = 2
        # except IndexError:
            # df['SEGMENT'][segments[j]] = 2

#load data from files into list of DataFrames
# dfu_list = list()

# for name in ['c:\data\segmentMzgR8e1.csv', 'c:\data\segmentGrnR3e2.csv', 'c:\data\segmentAlmR3e1.csv', 'c:\data\segmentNkgR9e3.csv', 'c:\data\segmentMzgR8e3.csv', 'c:\data\segmentPrkR1e11.csv', 'c:\data\segmentNkgR9e2.csv', 'c:\data\segmentGrnR6e1.csv', 'c:\data\segmentChaR6e2.csv', 'c:\data\segmentGrnR3e2.csv', 'c:\data\segmentNkgR9e3.csv']:
 # print(name)
 # dfu_buf = read_csv(name,';', index_col=["TIMESTAMP"], parse_dates=["TIMESTAMP"], dayfirst=True)
 # dfu_list.append(dfu_buf)
# for i in range(len(dfu_list)):
 # plt.plot(dfu_list[i].POUTB)
 # plt.plot(dfu_list[i].SEGMENT)
# plt.show()

# for i in range(len(dfu_list)-1):
 # dfu_list[i+1].POUTB = dfu_list[i+1].POUTB - (dfu_list[i+1].POUTB.min() - dfu_list[i].POUTB.min())
 # print i, i+1

# dfu = concat([dfu_list[i] for i in range(len(dfu_list))])
# dfu.to_csv('c:/data/segments.csv', sep=';')

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
 