import csv
import numpy
from numpy import *
import pdb
from sklearn.linear_model.base import LinearModel
from sklearn.linear_model import *
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
import copy
from copy import *
from StratifiedShuffleSplit import *
import textmining
import collections
from collections import *
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

def readData(filename):
	f = open(filename, "rb")
	fr = csv.reader( (line.replace('\0','') for line in f) )
	#no error checking for now	

	x = []
	y = []
	for data in fr:
		#convert data to numeric type
		#for i in range(len(data)):
		#  data[i] = float(data[i])
		#if data[470] == "1":
		x1=data[0:8]
		y.append(data[8])
		x.append(x1)
	f.close()
	return [x,y]

def readRawData(filename):
	f = open(filename, "rb")
	fr = csv.reader( (line.replace('\0','') for line in f) )
	vocab_x = {}
	vocab_y = {}
	tdx = textmining.TermDocumentMatrix()
	tdy = textmining.TermDocumentMatrix()

	# Find all words in abstract, and all labels in dataset
	for data in fr:
		tdx.add_doc(data[1])
		tdy.add_doc(" ".join(data[2:]))

	f.close()

	# print
	xrowsg = tdx.rows(cutoff=2)
	yrowsg = tdy.rows(cutoff=2)
	xrows = []
	yrows = []
	for row in xrowsg:
		xrows.append(row)
	for row in yrowsg:
		yrows.append(row)
	
	return [xrows, yrows]
	for i in range(len(xrows)):
		print str(xrows[i]) + "\t" + str(yrows[i])
		pdb.set_trace()

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_number_vector(s):
	for i in s:
		if not is_number(i): return False
	return True

def normalize(x, xref=[]):
  m = [] # mean
  s = [] # standard devs
  p = len(x[0]) # number of features

  retval = deepcopy(x)
  if xref == []: 
    xref = retval

  for i in range(p):
    featurei = array([xref[temp][i] for temp in range(len(xref))])
    m.append(featurei.mean())
    s.append(featurei.std())

  for i in range(len(x)):
    for j in range(p):
      retval[i][j] = ((1.0*x[i][j])-m[j])/s[j]
  return retval

def normalize_scale(x, xref = []):
	minval = [] # min of each predictor
	maxval = [] # max of each predictor
	p = len(x[0])

	retval = deepcopy(x)
	if xref == []:
		xref = retval
	
	# find the min and max values
	for i in range(p):
		featurei = array([xref[temp][i] for temp in range(len(xref))])
		minval.append(float(featurei.min()))
		maxval.append(float(featurei.max()))
		if minval[i] == maxval[i]:
			maxval[i] = minval[i]+1 # all values of this feature are 0, so this prevents div zero error
	
	for i in range(len(x)):
		for j in range(p):
			scaled_01 = (float(x[i][j]) - minval[j])/(maxval[j] - minval[j])
			retval[i][j] = (scaled_01 * 2) - 1.0
	return retval

def zerooneloss(y, yp):
	yp = array(yp)*1.0
	y = array(y)*1.0
	diff = abs(y - yp)
	n = len(y)*1.0
	z1 = sum(diff)/(2.0*n)
	print "\t\t Zeroone loss: " + str(z1)
	return z1 

def losses(y, yp):
	yp = array(yp)*1.0
	y = array(y)*1.0
	#print "\t\tComputing losses "
	z1 = zerooneloss(y, yp)
	pids = where(yp==1)[0]
	#print "\t\t Positive predictions: " + str(len(pids))

	if (len(pids)) == 0:
		precision = -1
	else:
		precision = len(where(y[pids]==1)[0])*(1.0)/len(pids)
	
	pids = where(y==1)[0]
	#print "\t\t Positive Instances: " + str(len(pids))
	if len(pids) == 0:
		recall = -1
	else:
		recall = len(where(yp[pids]==1)[0])*(1.0)/len(pids)

	return [z1, precision, recall]



def stratifier(ylabel, folds):
  n = len(ylabel)
  if len(shape(ylabel)) == 1:
    y = [[ylabel[i]] for i in range(n)]
  else:
    y = deepcopy(ylabel)
  k = len(y[0])
  ystrat = []
  strat = int(log2((1.0*n)/(folds))) # max tags possible for good stratification
  strat = min(strat, k)
  for i in range(n):
    yi_strat = 0 
    for j in range(strat):
      yi_strat = yi_strat + (2**(strat-1-j))*y[i][j]
    #print "Label: " + str(y[i]) + " stratified value: " + str(yi_strat)
    ystrat.append(yi_strat)

  #make sure ystrat has no labels occurring just once or twice (stratification error otherwise)
  ystrat = array(ystrat)
  ystratu = unique(ystrat)
  for i in range(len(ystratu)):
    ids=where(ystrat==ystratu[i])[0]
    if len(ids) < folds:
      print "Rare label! " + str(ystratu[i]) + " count: " + str(len(ids))
      ystrat[ids]=0
  ystratu = unique(ystrat)
  ids=where(ystrat==0)[0]
  if (len(ids) < folds and len(ids) > 0): ystrat[ids]=ystratu[1]

  return ystrat

