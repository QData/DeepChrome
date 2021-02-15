import torch
import collections
import pdb
import torch.utils.data
import csv
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import math
from pdb import set_trace as stop
import numpy as np



def loadData(filename,windows):
	with open(filename) as fi:
		csv_reader=csv.reader(fi)
		data=list(csv_reader)

		ncols=(len(data[0]))
	fi.close()
	nrows=len(data)
	ngenes=nrows/windows
	nfeatures=ncols-1
	print("Number of genes: %d" % ngenes)
	print("Number of entries: %d" % nrows)
	print("Number of HMs: %d" % nfeatures)

	count=0
	attr=collections.OrderedDict()

	for i in range(0,nrows,windows):
		hm1=torch.zeros(windows,1)
		hm2=torch.zeros(windows,1)
		hm3=torch.zeros(windows,1)
		hm4=torch.zeros(windows,1)
		hm5=torch.zeros(windows,1)
		for w in range(0,windows):
			hm1[w][0]=int(data[i+w][2])
			hm2[w][0]=int(data[i+w][3])
			hm3[w][0]=int(data[i+w][4])
			hm4[w][0]=int(data[i+w][5])
			hm5[w][0]=int(data[i+w][6])
		geneID=str(data[i][0].split("_")[0])

		thresholded_expr = int(data[i+w][7])

		attr[count]={
			'geneID':geneID,
			'expr':thresholded_expr,
			'hm1':hm1,
			'hm2':hm2,
			'hm3':hm3,
			'hm4':hm4,
			'hm5':hm5
		}
		count+=1

	return attr


class HMData(Dataset):
	# Dataset class for loading data
	def __init__(self,data_cell1,transform=None):
		self.c1=data_cell1
	def __len__(self):
		return len(self.c1)
	def __getitem__(self,i):
		final_data_c1=torch.cat((self.c1[i]['hm1'],self.c1[i]['hm2'],self.c1[i]['hm3'],self.c1[i]['hm4'],self.c1[i]['hm5']),1)
		label=self.c1[i]['expr']
		geneID=self.c1[i]['geneID']
		sample={'geneID':geneID,
			   'input':final_data_c1,
			   'label':label,
			   }
		return sample

def load_data(args):
	'''
	Loads data into a 3D tensor for each of the 3 splits.

	'''
	print("==>loading train data")
	cell_train_dict1=loadData(args.data_root+"/"+args.cell_type+"/classification/train.csv",args.n_bins)
	train_inputs = HMData(cell_train_dict1)

	print("==>loading valid data")
	cell_valid_dict1=loadData(args.data_root+"/"+args.cell_type+"/classification/valid.csv",args.n_bins)
	valid_inputs = HMData(cell_valid_dict1)

	print("==>loading test data")
	cell_test_dict1=loadData(args.data_root+"/"+args.cell_type+"/classification/test.csv",args.n_bins)
	test_inputs = HMData(cell_test_dict1)

	Train = torch.utils.data.DataLoader(train_inputs, batch_size=args.batch_size, shuffle=True)
	Valid = torch.utils.data.DataLoader(valid_inputs, batch_size=args.batch_size, shuffle=False)
	Test = torch.utils.data.DataLoader(test_inputs, batch_size=args.batch_size, shuffle=False)

	return Train,Valid,Test


