# -*- coding: cp866 -*-
"""
This script provides handling and analysis of GPP data
2014 Nataliya Stoyan
"""


import sys
import numpy as np
from pandas import *
from matplotlib import pyplot as plt
from scipy.cluster.vq import kmeans, vq, whiten, kmeans2
from pybrain.tools.shortcuts import buildNetwork 
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.structure.modules import SoftmaxLayer, SigmoidLayer, LinearLayer, TanhLayer 
from pybrain.tools.xml.networkwriter import NetworkWriter


# Time-series with crashes: exportGrnR3e2.csv, exportMzgR8e1.dsv, exportAlmR3e1.csv, exportNkgR9e3.dsv, 
# exportMzgR8e3.dsv, exportPrkR1e11.dsv, #exportVtkR1e2.dsv, exportNkgR9e2.сsv, 
# exportNkgR10e1.dsv, exportGrnR6e1.dsv, exportChaR6e2.dsv, exportIgrR1e1.dsv
# Time-series without crashes: crashes: exportGrnR2e4.dsv


# initializing tuple with parameters names for handling time-series
PARAMETERS = ("SD", "PINB", "TINB", "Q", "POUTB", "TOUTB", "REVHP",
    "REVLP", "REVWE", "TFIRE","CF"
)


# initializing tuple with events names for handling time-series
EVENTS = ("AVARIA", "REMONT", "RABOTA", "REZERV", "NOINFO")


# initializing dictionaries with average and minimum/maximum values of 
# time-series for normalization function
MEAN_VALUES = {"AVG(SD)":48.448, "AVG(PINB)": 46.105, "AVG(POUTB)": 56.413, 
    "AVG(TINB)": 14.538, "AVG(TOUTB)": 23.412, "AVG(Q)": 22197.743, "AVG(REVHP)": 5278.730, 
    "AVG(REVLP)": 3619.455, "AVG(REVWE)": 3506.259, "AVG(TFIRE)": 431.362, 
    "AVG(CF)": 1.438
}


MIN_MAX_VALUES = {"MAX-MIN(SD)": 225.9768, "MAX-MIN(PINB)": 693.7534, 
    "MAX-MIN(POUTB)": 693.7534, "MAX-MIN(TINB)": 6621.54029, 
    "MAX-MIN(TOUTB)": 6653.5, "MAX-MIN(Q)": 902902.1, 
    "MAX-MIN(REVHP)": 11618.02, "MAX-MIN(REVLP)": 10291.188,
    "MAX-MIN(REVWE)": 10000.0915555, "MAX-MIN(TFIRE)": 6728.5, 
    "MAX-MIN(CF)": 84.047723
}


def load_data(path):
    """ Load data from .csv file."""
    df = read_csv(
        path,';', index_col=["TIMESTAMP"],
        parse_dates=["TIMESTAMP"], dayfirst=True
        )
    print "data loaded, number of examples: %4d" %len(df)
    return df


def events_handling(df):
    """Replace character event labels with numbers."""
    print "events handling ... "
    for event in EVENTS:
        df[event] = 0 
    for i in xrange(len(df.index)):
        if df["EVENT"][i] == '\xc0\xe2\xe0\xf0\xe8\xff':#avaria
            df["AVARIA"][i] = 1
        if df["EVENT"][i] == '\xc2 \xf0\xe5\xec\xee\xed\xf2\xe5':#v remonte
            df["REMONT"][i] = 1
        if df["EVENT"][i] == '\xc2 \xf0\xe0\xe1\xee\xf2\xe5':#v rabote
            df["RABOTA"][i] = 1
        if df["EVENT"][i] == '\xc2 \xf0\xe5\xe7\xe5\xf0\xe2\xe5':#v rezerve
            df["REZERV"][i] = 1
        if df["EVENT"][i] == '\xcd\xe5\xf2 \xe4\xe0\xed\xed\xfb\xf5':#net dannyh
            df["NOINFO"][i] = 1
    del df["EVENT"]
    return df


def normalize_array(arr):
    """ Helper function for normalizing data in array. """
    arr = (arr - np.mean(arr))/(np.max(arr) - np.min(arr))
    return arr


def normalize_dataframe(df, param, mean, minmax):
    """ Helper function for normalizing data in dataframe. """
    print "normalizing data ... "
    df[param] = (
        (df[param] - mean[''.join(['AVG(',param,')'])]) / (
            minmax[''.join(['MAX-MIN(',param,')'])]
        )
    )
    return df


def normalize_data(df):
    """ Normalizing and scaling data. """
    print "normalizing data ..."
    # read dictionary with min/max values for normalization.
    for param in PARAMETERS:
        df = normalize_dataframe(df, param, MEAN_VALUES, MIN_MAX_VALUES)
        for i in xrange(1, 4):# some values are too small, they should be multiplied by 10 or 100
            if abs(df[param]).max() < 0.00009 * (10**i):
                df[param] = df[param] * 10**(4-i)
    return df


def make_equidistant(df):
    """ Make time-series in dataframe equidistant with 90 minutes interval."""
    print "makeing equidistant ... "
    df = df.resample('90Min')
    df = df.dropna(subset=PARAMETERS, how='all')
    df = df.ffill()
    print "data handled, number of examples: %4d" %len(df)
    return df


def crash_handling(df):
    """ Mark as crash all examples recorded within 48 hours before a crash. """
    print "CRASH events handling ... "
    df['CRASH'] = 0
    for i in xrange(len(df.index)):
        if df.AVARIA[i] <> 0:
            print "AVARIA handling"
            df.CRASH[i-32:i+1] = 1  
    return df


def clustering(df):
    print "clustering ..."
    dfc = df
    dfc = whiten(dfc) # sort of normalization
    data = np.ndarray((len(dfc.index), len(dfc.axes)), buffer = dfc[PARAMETERS].values)
    #print data
    #centroids, idx = kmeans2(data, 3)
    centroids, dis = kmeans(data, 3)
    idx, _ = vq(data, centroids)
    badcentroids = 0
    # check min cost function here
    plt.plot(data[idx==0,0],data[idx==0,1],'ob', 
         data[idx==1,0],data[idx==1,1],'or',
         data[idx==2,0],data[idx==2,1],'oy')
    plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.98,
                wspace=None, hspace=None)
    plt.show()
    return centroids, data, idx  


def plot_dataframe(df):
    print "plotting dataframe ..."
    for param in PARAMETERS:
        plt.plot(df.index, df[param], label = param)
    try:
        for i in xrange(len(df.index)):
            if df.CRASH[i] <> 0:
                plt.plot_date(df.index[i], df.CRASH[i], "rx")    
            else:
                plt.plot_date(df.index[i], df.REMONT[i], "bo")
    except: 
        print "No events information found"
    try:
        plt.plot(df.index, df.CLUSTER, "r--", label = "CLUSTER")
    except: 
        print "No claster information found"
    plt.ylim((-1.1,1.1))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=None, top=0.99,
                wspace=None, hspace=None)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=9, borderaxespad=0.)
    plt.show()


def plot_cluster(data, centroids, idx):
    print data
    print "plotting clusters ..."
    plt.plot(data[idx==0,0],data[idx==0,1],'ob',
             data[idx==1,0],data[idx==1,1],'or')
    plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.99,
                        wspace=None, hspace=None)
    plt.show()


def map_dataframe_and_clusters(df, idx):
    df["CLUSTER"] = norm(idx)
    #df['CLUSTER'] = idx
    return df


def create_dataset(df):
    """ Create dataset for neural network analysis. """
    print "creating dataset ..."
    ds = ClassificationDataSet(8, 1, nb_classes=2, class_labels=[
                               "crash", "nocrash"])
    for i in xrange(len(df.index)):
        buf = list()
        flag = 1
        for par in ["PINB", "POUTB", "TOUTB",
                    "REVHP","REVLP","REVWE","TFIRE", "CF"]:
            if np.isnan(df[par][i]): flag = 0
            buf.append(df[par][i])
        if flag == 1: 
            ds.addSample(buf, df.CRASH[i])
    number = len((np.nonzero(df.CRASH)[0]))
    length = len(ds)
    percent = float(number) / length*100
    print "number of examples: ", length
    print "number of positive examples: ", number
    print "percent of positive examples: ", percent
    # print "Data Set: ", ds
    return ds


def network_training(ds):
    """ Builiding and training network. """
    print "network training ..."
    tries = 2
    bias = True
    fast = False
    previous_error = 100
    epochs = 60
    layer_dim = 1
    for _ in xrange(tries):
        print " try: %4d" % _
        train_ds, test_ds = ds.splitWithProportion(0.7)
        try_net = buildNetwork(train_ds.indim, int(train_ds.indim*layer_dim), train_ds.outdim, hiddenclass=SigmoidLayer, outclass=SoftmaxLayer, bias=bias, fast=fast)
        trainer = BackpropTrainer(try_net, train_ds)
        trainer.trainEpochs(epochs)
        for mod in try_net.modules:
            print "Module:", mod.name
            if mod.paramdim > 0:
                print "--parameters:", mod.params
            for conn in try_net.connections[mod]:
                print "-connection to", conn.outmod.name
                if conn.paramdim > 0:
                    print "-parameters", conn.params
            if hasattr(try_net, "recurrentConns"):
                print "Recurrent connections"
                for conn in try_net.recurrentConns:             
                    print "-", conn.inmod.name, " to", conn.outmod.name
                    if conn.paramdim > 0:
                        print "- parameters", conn.params
        trnresult = percentError(trainer.testOnClassData(), train_ds['class'])
        tstresult = percentError(trainer.testOnClassData(dataset=test_ds ), test_ds['class'])
        #print test_ds['target']
        print "epoch: %4d" % trainer.totalepochs, \
        " train error: %5.2f%%" % trnresult, \
        " test error: %5.2f%%" % tstresult
        if tstresult < previous_error:
            net = try_net
            previous_error = tstresult
            NetworkWriter.writeToFile(net, "net.xml")
            layer_dim = layer_dim * 2


def main():
    df = load_data(sys.argv[1])
    plot_dataframe(df)
    df = events_handling(df)
    plot_dataframe(df)
    df = normalize_data(df)
    plot_dataframe(df)
    df = make_equidistant(df)
    plot_dataframe(df)
    df = crash_handling(df)
    plot_dataframe(df)
    #centroids, data, idx = clustering(df)
    #df = map_dataframe_and_clusters(df, idx)
    #plot_cluster(df, centroids, idx)
    ds = create_dataset(df)
    network_training(ds)


if __name__ == "__main__":
    main()