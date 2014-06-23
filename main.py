## -*- coding: cp1251 -*-

import numpy as np
import scipy as sp
from pandas import *
import statsmodels.api as sm
from matplotlib import pyplot as plt
# import pywt
from math import *
from pybrain.tools.shortcuts     import buildNetwork 
from pybrain.datasets            import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer, SigmoidLayer, LinearLayer, TanhLayer 
from pybrain.tools.xml.networkwriter import NetworkWriter

print('PREPARE DATA')
filepath = 'c:/data/'
print(''.join(['Current filepath is "',filepath,'". Do you want to change filepath? y/n']))
if raw_input() == 'y':
 print('Enter new filepath:')
 filepath = raw_input()

print('Enter filename:')
filename = raw_input()
print ('Enter delimiter:')
delimiter = raw_input()

#с авариями: exportGrnR3e2.csv, exportMzgR8e1.dsv, exportAlmR3e1.csv, exportNkgR9e3.dsv, exportMzgR8e3.dsv, exportPrkR1e11.dsv, 
#exportVtkR1e2.dsv, exportNkgR9e2.dsv (delimiter = ',''), exportNkgR10e1.dsv, exportGrnR6e1.dsv, c:\data\exportChaR6e2.dsv, exportIgrR1e1.dsv
#без аварий: exportGrnR2e4.dsv

df = read_csv(''.join([filepath,filename]),delimiter, index_col=["TIMESTAMP"], parse_dates=["TIMESTAMP"], dayfirst=True) 
df = read_csv('c:\data\export_all.dsv',';', index_col=["TIMESTAMP"], parse_dates=["TIMESTAMP"], dayfirst=True) 
df = read_csv('c:\data\exportIgrR1e1.dsv',';', index_col=["TIMESTAMP"], parse_dates=["TIMESTAMP"], dayfirst=True) 

#LABEL in df and WRITE FILE WITHOUT EQUID
df['LABEL']  = 0
for i in range(len(df)):
 if df['EVENT'][i] == '\xc0\xe2\xe0\xf0\xe8\xff':#авария
  df['LABEL'][i] = 1
 if df['EVENT'][i] == '\xc2 \xf0\xe5\xec\xee\xed\xf2\xe5':#ремонт
  df['LABEL'][i] = 1
 if df['EVENT'][i] == '\xc2 \xf0\xe0\xe1\xee\xf2\xe5':#работа
  df['LABEL'][i] = 1
 if df['EVENT'][i] == '\xc2 \xf0\xe5\xe7\xe5\xf0\xe2\xe5':#резерв
  df['LABEL'][i] = 1
 if df['EVENT'][i] == '\xcd\xe5\xf2 \xe4\xe0\xed\xed\xfb\xf5':#нет данных
  df['LABEL'][i] = 1
del df['EVENT'] 
df = df.ffill()

#reading dictionary with min/max values for normalization
def read_dict(path):
 dic={}
 for line in open(path):
  key,value=line.strip().split(':')
  dic[key]=float(value)  #dic[key]=value
 return dic
minmax_dict = read_dict(r'c:\data\minmax-sub.txt')
mean_dict = read_dict(r'c:\data\mean.txt')

df.SD = (df.SD - mean_dict['AVG(SD)']) / (minmax_dict['MAX-MIN(SD)'])
df.PINB = (df.PINB - mean_dict['AVG(PINB)']) / (minmax_dict['MAX-MIN(PINB)'])
df.POUTB = (df.POUTB - mean_dict['AVG(POUTB)']) / (minmax_dict['MAX-MIN(POUTB)'])
df.TINB = (df.TINB - mean_dict['AVG(TINB)']) / (minmax_dict['MAX-MIN(TINB)'])
df.TOUTB = (df.TOUTB - mean_dict['AVG(TOUTB)']) / (minmax_dict['MAX-MIN(TOUTB)'])
df.CF = (df.CF - mean_dict['AVG(CF)']) / (minmax_dict['MAX-MIN(CF)'])
df.TFIRE = (df.TFIRE - mean_dict['AVG(TFIRE)']) / (minmax_dict['MAX-MIN(TFIRE)'])
df.REVHP = (df.REVHP - mean_dict['AVG(REVHP)']) / (minmax_dict['MAX-MIN(REVHP)'])
df.REVLP = (df.REVLP - mean_dict['AVG(REVLP)']) / (minmax_dict['MAX-MIN(REVLP)'])
df.REVWE = (df.REVWE - mean_dict['AVG(REVWE)']) / (minmax_dict['MAX-MIN(REVWE)'])
df['SEGMENT'] = 0

#MANUAL SEGMENTATION
df['SEGMENT'][0] = 1
df['SEGMENT'][-1] = 1

if filename == 'exportGrnR3e2.csv': ####SEGMENTS FOR GrnR3e2  
 df.SEGMENT[3758*2] = 1
 df.SEGMENT[7133*2] = 1
 df.SEGMENT[14700*2] = 1

if filename == 'exportMzgR8e1.dsv':
 df.SEGMENT[156*2] = 1
 df.SEGMENT[248*2] = 1
 df.SEGMENT[336*2] = 1
 df.SEGMENT[665*2] = 1
 df.SEGMENT[758*2] = 1
 df.SEGMENT[1115*2] = 1
 df.SEGMENT[1580*2] = 1
 df.SEGMENT[5577*2] = 1
 df.SEGMENT[5913*2] = 1
 df.SEGMENT[6363*2] = 1
 df.SEGMENT[6419*2] = 1
 df.SEGMENT[6465*2] = 1


if filename == 'exportNkgR9e3.dsv':# ####SEGMENTS FOR NkgR9e3
 df.SEGMENT[4800] = 1
 df.SEGMENT[11367] = 1
 df.SEGMENT[11547] = 1
 df.SEGMENT[24602] = 1
 df.SEGMENT[28733] = 1
 df.SEGMENT[37759] = 1
 df.SEGMENT[37766] = 1


if filename == 'exportMzgR8e3.dsv':   # ####SEGMENTS FOR MzgR8e3
 df.SEGMENT[1060] = 1
 df.SEGMENT[1350] = 1
 df.SEGMENT[1680] = 1
 df.SEGMENT[1765] = 1
 df.SEGMENT[1950] = 1
 df.SEGMENT[2130] = 1
 df.SEGMENT[2390] = 1
 df.SEGMENT[2390] = 1

if filename == 'exportPrkR1e11.dsv':    ####SEGMENTS FOR PrkR1e11
 df.SEGMENT[376] = 1
 df.SEGMENT[416] = 1
 df.SEGMENT[999] = 1
 df.SEGMENT[1035] = 1
 df.SEGMENT[1300] = 1
 df.SEGMENT[1347] = 1
 df.SEGMENT[1660] = 1
 df['SEGMENT'][0] = 1
 df['SEGMENT'][-1] = 1

if filename == 'exportNkgR9e2.dsv':   ####SEGMENTS FOR NkgR9e2
 df.SEGMENT[610] = 1
 df.SEGMENT[1620] = 1

if filename == 'exportNkgR10e1.dsv':      ####SEGMENTS FOR NkgR10e1
 df.SEGMENT[1180] = 1
 df.SEGMENT[3600] = 1
 df.SEGMENT[3700] = 1
 df.SEGMENT[3910] = 1
 df.SEGMENT[4164] = 1
 df.SEGMENT[4360] = 1
 df.SEGMENT[4910] = 1


if filename == 'exportChaR6e2.dsv':  ####SEGMENTS FOR ChaR6e2
 df.SEGMENT[657] = 1
 df.SEGMENT[2840] = 1
 df.SEGMENT[7290] = 1


if filename == 'exportGrnR6e1.dsv':   ####SEGMENTS FOR GrnR6e1
 df.SEGMENT[592] = 1
 df.SEGMENT[803] = 1
 df.SEGMENT[1753] = 1
 df.SEGMENT[4077] = 1
 df.SEGMENT[4166] = 1
 df.SEGMENT[5705] = 1
 df.SEGMENT[11105] = 1
 df.SEGMENT[12395] = 1
 df.SEGMENT[12555] = 1
 df.SEGMENT[12374] = 1
 df.SEGMENT[13464] = 1
 df.SEGMENT[13464] = 1


#MARK SEGMENTS WITH ACCEDENTS

segments = list()
for i in range(len(df)):
 if df['SEGMENT'][i] == 1:
   segments.append(i)
for i in range(len(df)):
 if df['LABEL'][i] > 0.9:
  for j in range(len(segments)):
   if i > segments[j]:
    try:
     if i < segments[j+1]:
      df['SEGMENT'][segments[j]] = 2
    except IndexError:
     df['SEGMENT'][segments[j]] = 2

print('Data is loaded. Show plot? y/n')
if raw_input() == 'y':
 plt.plot(df.PINB, label = 'PinB')
 plt.plot(df.POUTB, label = 'PoutB')
 plt.plot(df.SD, label = 'SD')
 plt.plot(df.Q, label = 'Q')
 plt.plot(df.TINB, label = 'TinB')
 plt.plot(df.TOUTB, label = 'ToutB')
 plt.plot(df.REVWE, label = 'RevWE')
 plt.plot(df.REVHP, label = 'RevHP')
 plt.plot(df.REVLP, label = 'RevLP')
 plt.plot(df.TFIRE, label = 'Tfire')
 plt.plot(df.CF, label = 'CF')
 plt.plot(df.LABEL, label = 'Label')
 plt.plot(df.SEGMENT, label = 'Segment')
 plt.legend(bbox_to_anchor=(1.05, 1), loc=9, borderaxespad=0.)
 plt.show()
 
#UNIFICATION

indx = date_range('2013-02-20', periods=len(df), freq='10Min')
df.index = indx
ind = [i for i in range(len(df)) if df['SEGMENT'][i]>0.9]
labels = [i for i in range(len(df)) if df['SEGMENT'][i]>1]
df_list = list()
dfu = DataFrame()
for i in range(len(ind)-1):
 buf = df[ind[i]:ind[i+1]]
 print ind[i], ind[i+1]
 coef = 600/(float(600)/len(buf))
 buf = buf.asfreq(str(int(coef-1))+'S', method = 'pad')
 buf = buf[:600]
 df_list.append(buf) 
 
dfu = concat(df_list)

i = 0
while i in range(len(dfu) - 100):
  if dfu['SEGMENT'][i] > 0.9:
   print i
   j = 1
   while j < 100:
    dfu['SEGMENT'][i+j] = 0
    j += 1
   i = i + j
  else:
   i += 1
print('Segments are unificated. Show plot? y/n')
if raw_input() == 'y':
 plt.plot(dfu.PINB, label = 'PinB')
 plt.plot(dfu.POUTB, label = 'PoutB')
 plt.plot(dfu.SD, label = 'SD')
 plt.plot(dfu.Q, label = 'Q')
 plt.plot(dfu.TINB, label = 'TinB')
 plt.plot(dfu.TOUTB, label = 'ToutB')
 plt.plot(dfu.REVWE, label = 'RevWE')
 plt.plot(dfu.REVHP, label = 'RevHP')
 plt.plot(dfu.REVLP, label = 'RevLP')
 plt.plot(dfu.TFIRE, label = 'Tfire')
 plt.plot(dfu.CF, label = 'CF')
 plt.plot(dfu.LABEL, label = 'Label')
 plt.plot(dfu.SEGMENT, label = 'Segment')
 plt.legend(bbox_to_anchor=(1.05, 1), loc=9, borderaxespad=0.)
 plt.show()

print(''.join(['Write file ', filepath, 'segment', filename[6:13], '.csv? y/n']))
if raw_input() == 'y':
 dfu.to_csv(''.join([filepath, 'segment', filename[6:13], '.csv']), sep=';')


#load data from files into list of DataFrames
dfu_list = list()

for name in ['c:\data\segmentMzgR8e1.csv', 'c:\data\segmentGrnR3e2.csv', 'c:\data\segmentAlmR3e1.csv', 'c:\data\segmentNkgR9e3.csv', 'c:\data\segmentMzgR8e3.csv', 'c:\data\segmentPrkR1e11.csv', 'c:\data\segmentNkgR9e2.csv', 'c:\data\segmentGrnR6e1.csv', 'c:\data\segmentChaR6e2.csv', 'c:\data\segmentGrnR3e2.csv', 'c:\data\segmentNkgR9e3.csv']:
 print(name)
 dfu_buf = read_csv(name,';', index_col=["TIMESTAMP"], parse_dates=["TIMESTAMP"], dayfirst=True)
 dfu_list.append(dfu_buf)
for i in range(len(dfu_list)):
 plt.plot(dfu_list[i].POUTB)
 plt.plot(dfu_list[i].SEGMENT)
plt.show()
#
for i in range(len(dfu_list)-1):
 dfu_list[i+1].POUTB = dfu_list[i+1].POUTB - (dfu_list[i+1].POUTB.min() - dfu_list[i].POUTB.min())
 print i, i+1

dfu = concat([dfu_list[i] for i in range(len(dfu_list))])
dfu.to_csv('c:/data/segments.csv', sep=';')

#NEURAL NETWORK
#Teaching and testing network
dfu = read_csv('c:/data/segments.csv',';', index_col=["TIMESTAMP"], parse_dates=["TIMESTAMP"], dayfirst=True)
ind = [i for i in range(len(dfu)) if dfu['SEGMENT'][i]>0.9]
labels = list()

for i in range(len(ind)):
 if dfu['SEGMENT'][ind[i]]>1: 
  j = 1 
 else: 
  j = 0
 labels.append(j)
 
ds = ClassificationDataSet(600, 1, nb_classes=2, class_labels=['bad', 'good']) 
#add to data set segments without empty values
for i in range(len(ind)):
 j = ind[i]
 flag = 1
 while j in range(ind[i], ind[i]+600):
  if isnan(dfu.CF[j]):
   flag = 0
  j+=1
 if flag:
  ds.addSample(dfu.CF[ind[i]:(ind[i]+600)], (labels[i], ))

for i in range(len(ind)):
 j = ind[i]
 flag = 1
 while j in range(ind[i], ind[i]+600):
  if isnan(dfu.SD[j]):
   flag = 0
  j+=1
 if flag:
  ds.addSample(dfu.SD[ind[i]:(ind[i]+600)], (labels[i], ))
  
for i in range(len(ind)):
 j = ind[i]
 flag = 1
 while j in range(ind[i], ind[i]+600):
  if isnan(dfu.TFIRE[j]):
   flag = 0
  j+=1
 if flag:
  ds.addSample(dfu.TFIRE[ind[i]:(ind[i]+600)], (labels[i], ))
  
for i in range(len(ind)):
 j = ind[i]
 flag = 1
 while j in range(ind[i], ind[i]+600):
  if isnan(dfu.REVLP[j]):
   flag = 0
  j+=1
 if flag:
  ds.addSample(dfu.REVLP[ind[i]:(ind[i]+600)], (labels[i], ))
  
for i in range(len(ind)):
 j = ind[i]
 flag = 1
 while j in range(ind[i], ind[i]+600):
  if isnan(dfu.REVWE[j]):
   flag = 0
  j+=1
 if flag:
  ds.addSample(dfu.REVWE[ind[i]:(ind[i]+600)], (labels[i], ))
  
for i in range(len(ind)):
 j = ind[i]
 flag = 1
 while j in range(ind[i], ind[i]+600):
  if isnan(dfu.REVHP[j]):
   flag = 0
  j+=1
 if flag:
  ds.addSample(dfu.REVHP[ind[i]:(ind[i]+600)], (labels[i], ))

for i in range(len(ind)):
 j = ind[i]
 flag = 1
 while j in range(ind[i], ind[i]+600):
  if isnan(dfu.Q[j]):
   flag = 0
  j+=1
 if flag:
  ds.addSample(dfu.Q[ind[i]:(ind[i]+600)], (labels[i], ))

for i in range(len(ind)):
 j = ind[i]
 flag = 1
 while j in range(ind[i], ind[i]+600):
  if isnan(dfu.TOUTB[j]):
   flag = 0
  j+=1
 if flag:
  ds.addSample(dfu.TOUTB[ind[i]:(ind[i]+600)], (labels[i], ))

for i in range(len(ind)):
 j = ind[i]
 flag = 1
 while j in range(ind[i], ind[i]+600):
  if isnan(dfu.POUTB[j]):
   flag = 0
  j+=1
 if flag:
  ds.addSample(dfu.POUTB[ind[i]:(ind[i]+600)], (labels[i], ))

for i in range(len(ind)):
 j = ind[i]
 flag = 1
 while j in range(ind[i], ind[i]+600):
  if isnan(dfu.PINB[j]):
   flag = 0
  j+=1
 if flag:
  ds.addSample(dfu.PINB[ind[i]:(ind[i]+600)], (labels[i], ))

for i in range(len(ind)):
 j = ind[i]
 flag = 1
 while j in range(ind[i], ind[i]+600):
  if isnan(dfu.TINB[j]):
   flag = 0
  j+=1
 if flag:
  ds.addSample(dfu.TINB[ind[i]:(ind[i]+600)], (labels[i], ))
  
tries = 10
bias = True
fast = False
previous_error = 100
epochs = 60
layer_dim = 0.1
for _ in xrange(tries):
 train_ds, test_ds = ds.splitWithProportion(0.7)
 try_net = buildNetwork(train_ds.indim, int(train_ds.indim*layer_dim), train_ds.outdim, hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=bias, fast=fast)
 trainer = BackpropTrainer(try_net, train_ds)
 trainer.trainEpochs(epochs)
 trnresult = percentError(trainer.testOnClassData(), train_ds['class'])
 tstresult = percentError(trainer.testOnClassData(dataset=test_ds ), test_ds['class'])
 print test_ds['target']
 print "epoch: %4d" % trainer.totalepochs, \
  " train error: %5.2f%%" % trnresult, \
  " test error: %5.2f%%" % tstresult
 if tstresult < previous_error:
            net = try_net
            previous_error = tstresult

 NetworkWriter.writeToFile(net, 'net.xml')
 layer_dim = layer_dim * 2

train_ds, test_ds = ds.splitWithProportion(0.7)
try_net = buildNetwork(train_ds.indim, train_ds.indim*2, train_ds.outdim, hiddenclass=TanhLayer, outclass=SoftmaxLayer, bias=bias,
 fast=fast)
trainer = BackpropTrainer(try_net, train_ds)
trainer.trainEpochs(epochs)
trnresult = percentError(trainer.testOnClassData(), train_ds['class'])
tstresult = percentError(trainer.testOnClassData(dataset=test_ds ), test_ds['class'])
print test_ds['target']
print "epoch: %4d" % trainer.totalepochs, \
 " train error: %5.2f%%" % trnresult, \
 " test error: %5.2f%%" % tstresult
 
################################################################################
#SD      PINB  TINB   Q  POUTB     TOUTB     REVHP REVLP     REVWE     TFIRE        CF


####WAVELETS
#WAVELET TRANSFORMATION
# cpinbA, cpinbD = pywt.dwt(df.PINB, 'db4')
# ctinbA, ctinbD = pywt.dwt(df.TINB, 'db4')
# cpoutbA, cpoutbD = pywt.dwt(df.POUTB, 'db4')
# ctoutbA, ctoutbD = pywt.dwt(df.TOUTB, 'db4')
# crevhpA, crevhpD = pywt.dwt(df.REVHP, 'db4')
# crevlpA, crevlpD = pywt.dwt(df.REVLP, 'db4')
# crevweA, crevweD = pywt.dwt(df.REVWE, 'db4')
# csdA, csdD = pywt.dwt(df.SD, 'db4')
# cqA, cqD = pywt.dwt(df.Q, 'db4')
# ctfireA, ctfireD = pywt.dwt(df.TFIRE, 'db4')
# ccfA, ccfD = pywt.dwt(df.CF, 'db4')
# d = {'cpinbA': cpinbA, 'cpinbD': cpinbD, 'cpoutbA': cpoutbA, 'cpoutbD': cpoutbD,  'ctinbA': ctinbA, 'ctinbD': ctinbD,'ctoutbA': ctoutbA, 'ctoutbD': ctoutbD, 'crevhpA': crevhpA, 'crevhpD': crevhpD, 'crevlpA': crevlpA, 'crevlpD': crevlpD, 'crevweA': crevweA, 'crevweD': crevweD, 'csdA': csdA, 'csdD': csdD, 'ctfireA': ctfireA, 'ctfireD': ctfireD, 'ccfA': ccfA, 'ccfD': ccfD, 'cqA': cqA, 'cqD': cqD} 
# dfw = DataFrame(d)
#dfw = (dfw - dfw.mean()) / (dfw.max() - dfw.min()) #normalization



#WAVELET SEGMENTATION
# dfw['SEGMENT'] = 0
# dfw['LABEL'] = 0
# for i in range (len(dfw)):
 # if abs(dfw['cpinbD'][i]) > 0.06 or abs(dfw['cpoutbD'][i]) > 0.06 or abs(dfw['ctinbD'][i]) > 0.06 or abs(dfw['ctoutbD'][i]) > 0.06 or abs(dfw['crevhpD'][i]) > 0.06 or abs(dfw['crevlpD'][i]) > 0.06 or abs(dfw['crevweD'][i]) > 0.06 or abs(dfw['cqD'][i]) > 0.06 or abs(dfw['ctfireD'][i]) > 0.06 or abs(dfw['ccfD'][i]) > 0.06:
  # dfw['SEGMENT'][i] = 1
# #or abs(dfw['csdD'][i]) > 0.06
# dfw['SEGMENT'][0] = 1
# dfw['SEGMENT'][-1] = 1

# #deleting excess segment labels
# i = 0
# while i in range(len(dfw) - 20):
  # if dfw['SEGMENT'][i] == 1:
   # print i
   # j = 1
   # while j < 20:
    # dfw['SEGMENT'][i+j] = 0
    # j += 1
   # i = i + j
  # else:
   # i += 1

####SEGMENTS FOR GrnR3e2  
# dfw.SEGMENT[3758] = 1
# dfw.SEGMENT[7133] = 1
# dfw.SEGMENT[14700] = 1
# dfw['SEGMENT'][0] = 1
# dfw['SEGMENT'][-1] = 1

####SEGMENTS FOR GrnR3e2  
# df.SEGMENT[3758*2] = 1
# df.SEGMENT[7133*2] = 1
# df.SEGMENT[14700*2] = 1
# df['SEGMENT'][0] = 1
# df['SEGMENT'][-1] = 1

####SEGMENTS FOR AlmR3e1
# dfw.SEGMENT[426] = 1
# dfw.SEGMENT[468] = 1
# dfw.SEGMENT[560] = 1
# dfw.SEGMENT[670] = 1
# dfw.SEGMENT[691] = 1
# dfw.SEGMENT[755] = 1
# dfw['SEGMENT'][0] = 1
# dfw['SEGMENT'][-1] = 1

###SEGMENTS FOR AlmR3e1
# df.SEGMENT[426*2] = 1
# df.SEGMENT[468*2] = 1
# df.SEGMENT[560*2] = 1
# df.SEGMENT[670*2] = 1
# df.SEGMENT[691*2] = 1
# df.SEGMENT[755*2] = 1
# df['SEGMENT'][0] = 1
# df['SEGMENT'][-1] = 1

####SEGMENTS FOR MzgR8e1
# dfw.SEGMENT[156*2] = 1
# dfw.SEGMENT[248*2] = 1
# dfw.SEGMENT[336*2] = 1
# dfw.SEGMENT[665*2] = 1
# dfw.SEGMENT[758*2] = 1
# dfw.SEGMENT[1115*2] = 1
# dfw.SEGMENT[1580*2] = 1
# dfw.SEGMENT[5577*2] = 1
# dfw.SEGMENT[5913*2] = 1
# dfw.SEGMENT[6363*2] = 1
# dfw.SEGMENT[6419*2] = 1
# dfw.SEGMENT[6465*2] = 1
# dfw['SEGMENT'][0] = 1
# dfw['SEGMENT'][-1] = 1


# #SAVE LABELS
# for i in range(len(df)):
 # if df['LABEL'][i] > 0.9:
   # dfw.LABEL[i/2] = 1
   
#MARK SEGMENTS WITH ACCEDENTS

# for i in range(len(dfw)):
 # if dfw['SEGMENT'][i] == 1:
   # segments.append(i)
# for i in range(len(dfw)):
 # if dfw['LABEL'][i] == 1:
  # for j in range(len(segments)):
   # if i > segments[j]:
    # try:
     # if i < segments[j+1]:
      # dfw['SEGMENT'][segments[j]] = 2
    # except IndexError:
     # dfw['SEGMENT'][segments[j]] = 2
	 
#SHOW PLOTS
# plt.plot(dfw['cpoutbA'], label = 'PoutB') 
# plt.plot(dfw['cpinbA'], label = 'PinB')
# plt.plot(dfw['cqA'], label = 'Q')
# plt.plot(dfw['csdA'], label = 'SD')
# plt.plot(dfw['ctinbA'], label = 'TinB')
# plt.plot(dfw['ctoutbA'], label = 'ToutB')
# plt.plot(dfw['crevweA'], label = 'RevWE')
# plt.plot(dfw['crevhpA'], label = 'RevHP')
# plt.plot(dfw['crevlpA'], label = 'RevLP')
# plt.plot(dfw['ctfireA'], label = 'Tfire')
# plt.plot(dfw.ccfA, label = 'CF')
# plt.plot(dfw['SEGMENT'], label = 'Segment')
# plt.plot(dfw['LABEL'], label = 'Label')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=9, borderaxespad=0.)
# plt.show()

# plt.plot(dfu['cpoutbA'], label = 'PoutB') 
# plt.plot(dfu['cpinbA'], label = 'PinB')
# plt.plot(dfu['cqA'], label = 'Q')
# plt.plot(dfu['csdA'], label = 'SD')
# plt.plot(dfu['ctinbA'], label = 'TinB')
# plt.plot(dfu['ctoutbA'], label = 'ToutB')
# plt.plot(dfu['crevweA'], label = 'RevWE')
# plt.plot(dfu['crevhpA'], label = 'RevHP')
# plt.plot(dfu['crevlpA'], label = 'RevLP')
# plt.plot(dfu['ctfireA'], label = 'Tfire')
# plt.plot(dfu.ccfA, label = 'CF')
# plt.plot(dfu['SEGMENT'], label = 'Segment')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=9, borderaxespad=0.)
# plt.show()