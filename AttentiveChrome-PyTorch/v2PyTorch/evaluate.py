import numpy
import torch
import scipy 
import scipy.sparse as sp
import logging
from six.moves import xrange
from collections import OrderedDict
import sys
import pdb
from sklearn import metrics
import torch.nn.functional as F
from torch.autograd import Variable
from pdb import set_trace as stop


def compute_aupr(all_targets,all_predictions):
    aupr_array = []
    for i in range(all_targets.shape[1]):
        try:
            precision, recall, thresholds = metrics.precision_recall_curve(all_targets[:,i], all_predictions[:,i], pos_label=1)
            auPR = metrics.auc(recall,precision)#,reorder=True)
            if not math.isnan(auPR):
                aupr_array.append(numpy.nan_to_num(auPR))
        except: 
            pass
    
    aupr_array = numpy.array(aupr_array)
    mean_aupr = numpy.mean(aupr_array)
    median_aupr = numpy.median(aupr_array)
    var_aupr = numpy.var(aupr_array)
    return mean_aupr,median_aupr,var_aupr,aupr_array


def compute_auc(all_targets,all_predictions):
    auc_array = []

    for i in range(all_targets.shape[1]):
        try:  
            auROC = metrics.roc_auc_score(all_targets[:,i], all_predictions[:,i])
            auc_array.append(auROC)
        except ValueError:
            pass
    
    auc_array = numpy.array(auc_array)
    mean_auc = numpy.mean(auc_array)
    median_auc = numpy.median(auc_array)
    var_auc = numpy.var(auc_array)
    return mean_auc,median_auc,var_auc,auc_array


def compute_metrics(predictions, targets):


	pred=predictions.numpy()
	targets=targets.numpy()

	mean_auc,median_auc,var_auc,auc_array = compute_auc(targets,pred)
	mean_aupr,median_aupr,var_aupr,aupr_array = compute_aupr(targets,pred)


	return mean_aupr,mean_auc



