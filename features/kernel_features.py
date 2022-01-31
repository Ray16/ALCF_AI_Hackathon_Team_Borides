#Module base
#Created by Aria Coraor

import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import seaborn as sns
import os



def main():
	""" Process sequence at args.file based on kernel set. 
	Output as a numpy column."""
		
	#Load all sequence and period data
	seqs,chis,pers = load_data()
	
	#Calculate exponential kernels, from both ends.
	exp_ks = calc_exp_ks(seqs)
	#Calculate sinusoidal/periodic kernels
	cos_ks = calc_cos_ks(seqs)

	#Save data
	save_data(exp_ks,cos_ks)
	print("Kernel calculation complete. Exiting...")

def load_data():
	"""Load the data of all datasets into numpy arrays.
	Returns:
		seqs: *list of 2d np.array of int*
			list of each dataset, loaded separately, only the
			sequences. 
		chis: *list of 1d np.array of float*
			list of interaction parameters loaded separately.
		pers: *list of 1d np.array of float*
			list of period lengths loaded separately.
		
	"""
	d_path = "./data/"
	dat_a = np.loadtxt(d_path + "dataSetA")
	dat_b = np.loadtxt(d_path + "dataSetB")
	dat_c = np.loadtxt(d_path + "dataSetC")
	dat_d = np.loadtxt(d_path + "dataSetD")
	data = [dat_a, dat_b, dat_c, dat_d]
	#Process into seqs, chis, pers
	seqs = [ele[:,1:-1].astype(int) for ele in data]
	chis = [ele[:,0] for ele in data]
	pers = [ele[:,-1] for ele in data]

	return seqs,chis,pers

def calc_exp_ks(seqs):
	"""Calculate squared exponential kernel values for all sequences in 
	each of data. Do not mix datasets. Span over a range of kernel widths,
	from stdev = 1 through stdev = 15. Apply to both beginning and end.
	Parameters:
				seqs: *list of 2d np.array of int*
						list of each dataset, loaded separately, only the
						sequences.	
	Returns:
		exp_ks: *list of 2d np.array*
			List whose elements correspond to those of data. Each
			sub-array is of shape [n_rows,30], where each row
			corresponds to a single simulation, and each column
			corresponds to a single exponential rate.
	"""
	
	#Calculate exponential kernel arrays
	dists = np.arange(32,dtype=float)
	stdevs = np.arange(1,16)
	kernels = np.zeros((15,32))
	#Set up reversed kernels
	rev_kernels = np.zeros((15,32))
	rev_dists = 31 - dists
	
	for i in range(15):
		kernels[i] = np.exp(-1./2. * dists**2/stdevs[i]**2)
		rev_kernels[i] = np.exp(-1./2. * rev_dists**2/stdevs[i]**2)
		#Normalize
		kernels[i] /= np.sum(kernels[i])
		rev_kernels[i] /= np.sum(rev_kernels[i])
	colors = cm.viridis
	plt.clf()
	for i in range(15):
		plt.plot(dists,kernels[i], color = colors(i/14.), label = "k = %s" % str(round(stdevs[i],1)))
	plt.xlabel("Monomer index")
	plt.ylabel("Kernel function value")
	plt.title("Exponential Kernels")
	plt.legend(loc="best", edgecolor='grey')
	plt.savefig("exp_kerns.png",dpi=600)

	#Combine forward and reverse kernels
	kernels = np.concatenate((kernels, rev_kernels), axis=0)
	kernels = np.matrix(kernels).T

	#For each sequence, calculate projection onto kernels
	exp_ks = []
	
	for i, seq in enumerate(seqs):
		dotp = 0. #Reset dotp pointer
		#Shape: Hold all kernel values for each sequence
		dotp = np.matmul(seq,kernels)
		exp_ks.append(dotp)
		
	return exp_ks

def calc_cos_ks(seqs):
	"""Calculate cos values for all sequences in each of data, of the form
	cos(dists*np.pi / k). Do not mix datasets. Span over a range of kernel 
	widths, from k = 1 to k =15.
	Parameters:
		seqs: *list of 2d np.array of int*
                        list of each dataset, loaded separately, only the
                                 sequences.
	Returns:
		cos_ks: *list of 2d np.array*
                        List whose elements correspond to those of data. Each
                        sub-array is of shape [n_rows,15], where each row
                        corresponds to a single simulation, and each column
                        corresponds to a single exponential rate.
	"""
	#Calculate cos kernel arrays
	dists = np.arange(32,dtype=float)
	ks = np.arange(1,16)
	kernels = np.zeros((15,32))
	#Set up reversed kernels
	
	for i in range(15):
		kernels[i] = np.cos(dists * np.pi / ks[i])
		#We don't normalize the cosine kernel
		#kernels[i] /= np.sum(kernels[i])
	
	#Combine forward and reverse kernels
	plt.clf()
	colors = cm.viridis
	for i in range(15):
		plt.plot(dists,kernels[i], color = colors(i/14.),label="k = %s" % str(round(ks[i],1)))

	plt.xlabel("Monomer index")
	plt.ylabel("Kernel function value")
	plt.title("Cosine Kernels")
	plt.legend(loc="best",edgecolor='grey')
	plt.savefig("cos_kerns.png",dpi=600)
	kernels = np.matrix(kernels).T
	#For each sequence, calculate projection onto kernels
	cos_ks = []
	
	for i, seq in enumerate(seqs):
		dotp = 0. #Reset dotp pointer
		#Shape: Hold all kernel values for each sequence
		dotp = np.matmul(seq,kernels)
		cos_ks.append(dotp)
		
	return cos_ks
	

def save_data(exp_ks,cos_ks):
	"""Concatenate the exponential and cosine kernels, and save together 
	for each data file.
	Parameters:
		exp_ks: *list of 2d np.array*
                        List whose elements correspond to those of data. Each
                        sub-array is of shape [n_rows,30], where each row
                        corresponds to a single simulation, and each column
                        corresponds to a single exponential rate.
		cos_ks: *list of 2d np.array*
                        List whose elements correspond to those of data. Each
                        sub-array is of shape [n_rows,15], where each row
                        corresponds to a single simulation, and each column
                        corresponds to a single exponential rate.
	"""
	chars = ["A","B","C","D"]
	for i in range(len(exp_ks)):
		#Concatenate exp and cos
		joined = np.concatenate((exp_ks[i],cos_ks[i]),axis=1)
		fn = "dataSet%s.kern" % chars[i]
		np.savetxt(fn,joined)
	print("Saved all kernels at dataSet<char>.kern.")
	pass

def load_outs():
	"""Load the ouput data and return it."""
	outs = []
	for char in ["A","B","C","D"]:
		outs.append(np.loadtxt("dataSet%s.kern" % char))
	return outs

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	args = parser.parse_args()
	
	main()