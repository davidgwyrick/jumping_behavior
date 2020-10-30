#Misc
import pdb,glob,fnmatch, sys
import os, time, datetime
import glob, fnmatch

#Base
import numpy as np
import pandas as pd
import scipy.stats as st
import multiprocessing as mp

#Plot
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as colors
from matplotlib import pyplot as plt

#Plotting Params
sns.set_style("ticks")
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.major.pad']='10'

color_names=['windows blue','red','amber','faded green','dusty purple','orange','steel blue','pink',
             'greyish','mint','clay','light cyan','forest green','pastel purple','salmon','dark brown',
             'lavender','pale green','dark red','gold','dark teal','rust','fuchsia','pale orange','cobalt blue']
color_palette = sns.xkcd_palette(color_names)
cc = sns.xkcd_palette(color_names)

#Decoding
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
#Decoding Params
nProcesses = 25
nShuffles = 100

##====================##
##===== Decoding =====##

def decode_labels(X,Y,train_index,test_index,classifier='LDA',clabels=None,X_test=None,Y_test=None,shuffle=True,classifier_kws=None):
    
    #Copy training index for shuffle decoding
    train_index_sh = train_index.copy()
    np.random.shuffle(train_index_sh)
    
    #Split data into training and test sets
    if X_test is None:
        #Training and test set are from the same time interval
        X_train = X[train_index,:]
        X_test = X[test_index,:]
        
        #Get class labels
        Y_train = Y[train_index]
        Y_test = Y[test_index]
    
    else:
        #Training and test set are from different epochs
        X_train = X
        Y_train = Y
     
    #How many classes are we trying to classify?
    class_labels,nTrials_class = np.unique(Y,return_counts=True)
    nClasses = len(class_labels)

    #Initialize Classifier
    if classifier == 'LDA':
        clf = LinearDiscriminantAnalysis()
    elif classifier == 'SVM':
        clf = svm.LinearSVC(max_iter=1E6)
    elif classifier ==  'MLP':
        clf = MLPClassifier(alpha=0.01, max_iter=10000)

    #Luca's decoder 
    if (classifier == 'Euclidean_Dist') | (classifier == 'nearest_neighbor'):
        nTrials_test, nNeurons = X_test.shape
        PSTH_train = np.zeros((nClasses,nNeurons))
        #Calculate PSTH templates from training data
        for iStim, cID in enumerate(class_labels):
            pos = np.where(Y_train == cID)[0]
            PSTH_train[iStim] = np.mean(X_train[pos],axis=0)
        
        Y_hat = np.zeros((nTrials_test,),dtype=int)
        for iTrial in range(nTrials_test):
            if classifier == 'Euclidean_Dist':
                #Predict test data by taking the minimum euclidean distance
                dist = [np.sum((X_test[iTrial] - PSTH_train[iStim])**2) for iStim in range(nClasses)]
                Y_hat[iTrial] =  class_labels[np.argmin(dist)]
            else:
                #Predict test data by taking the maximum correlation between the test population vector and training PSTHs
                Rs = [np.corrcoef(PSTH_train[iStim],X_test[iTrial])[0,1] for iStim in range(nClasses)]
                Y_hat[iTrial] =  class_labels[np.argmax(Rs)]
                
            #Just to check whether the classifier correctly decodes the training data
#         nTrials_train = X_train.shape[0]
#         Y_hat_train = np.zeros((nTrials_train,),dtype=int)
#         for iTrial in range(nTrials_train):
#             if classifier == 'Euclidean_Dist':
#                 #Predict test data by taking the minimum euclidean distance
#                 dist = [np.sum((X_train[iTrial] - PSTH_train[iStim])**2) for iStim in range(nClasses)]
#                 Y_hat_train[iTrial] =  class_labels[np.argmin(dist)]
#             else:
#                 #Predict test data by taking the maximum correlation between the test population vector and training PSTHs
#                 Rs = [np.corrcoef(PSTH_train[iStim],X_train[iTrial])[0,1] for iStim in range(nClasses)]
#                 Y_hat_train[iTrial] =  class_labels[np.argmax(Rs)]
#         pdb.set_trace()
    #All other classifiers
    else:
        #Fit model to the training data
        clf.fit(X_train, Y_train)

        #Predict test data
        Y_hat = clf.predict(X_test)
        Y_hat_train = clf.predict(X_train)
    
    #Calculate confusion matrix
    kfold_hits = confusion_matrix(Y_test,Y_hat,labels=clabels)
#     kfold_hits_train = confusion_matrix(Y_train,Y_hat_train,labels=clabels)
#     print(kfold_hits)
#     coef = clf.coef_
#     pdb.set_trace()

    ##===== Perform Shuffle decoding =====##
    if shuffle:
        kfold_shf = np.zeros((nShuffles,nClasses,nClasses))
        #Classify with shuffled dataset
        for iS in range(nShuffles):
            #Shuffle training indices
            np.random.shuffle(train_index_sh)
            Y_train_sh = Y[train_index_sh]

            #Initialize Classifier
            if classifier == 'LDA':
                clf_shf = LinearDiscriminantAnalysis()
            elif classifier == 'SVM':
                clf_shf = svm.LinearSVC(max_iter=1E6) #C=classifier_kws['C']
            elif classifier ==  'MLP':
                clf_shf = MLPClassifier(alpha=0.01, max_iter=10000)
            
            #"Luca's" decoder 
            if (classifier == 'Euclidean_Dist') | (classifier == 'nearest_neighbor'):
                nTrials_test, nNeurons = X_test.shape
                PSTH_sh = np.zeros((nClasses,nNeurons))
                
                #Calculate PSTH templates from training data
                for iStim, cID in enumerate(class_labels):
                    pos = np.where(Y_train_sh == cID)[0]
                    PSTH_sh[iStim] = np.mean(X_train[pos],axis=0)

                Y_hat = np.zeros((nTrials_test,),dtype=int)
                for iTrial in range(nTrials_test):
                    if classifier == 'Euclidean_Dist':
                        #Predict test data by taking the minimum euclidean distance
                        dist = [np.sum((X_test[iTrial] - PSTH_sh[iStim])**2) for iStim in range(nClasses)]
                        Y_hat[iTrial] =  class_labels[np.argmin(dist)]
                    else:
                        #Predict test data by taking the maximum correlation between the test population vector and training PSTHs
                        Rs = [np.corrcoef(PSTH_sh[iStim],X_test[iTrial])[0,1] for iStim in range(nClasses)]
                        Y_hat[iTrial] =  class_labels[np.argmax(Rs)]
                    
            #All other classifiers
            else:
                #Fit model to the training data
                clf_shf.fit(X_train, Y_train_sh)

                #Predict test data
                Y_hat = clf_shf.predict(X_test)

            #Calculate confusion matrix
            kfold_shf[iS] = confusion_matrix(Y_test,Y_hat,labels=clabels)
    else:
        kfold_shf = np.zeros((nClasses,nClasses))

        #Return decoding results
    return kfold_hits, kfold_shf

def calculate_accuracy(results,method='L1O',plot_shuffle=False,pdfdoc=None):
    
    nClasses = results[0][0].shape[0]
    #Save results to these
    confusion_mat = np.zeros((nClasses,nClasses))
    confusion_shf = np.zeros((nClasses,nClasses))
    confusion_z = np.zeros((nClasses,nClasses))
    
    if method == 'L1O':    
        c_shf = np.zeros((nShuffles,nClasses,nClasses))
        for iK,rTuple in enumerate(results):
            loo_hits = rTuple[0] #size [nClasses x nClasses]
            loo_shf = rTuple[1]  #size [nShuffles,nClasses x nClasses]

            #Add hits to confusion matrix
            confusion_mat += loo_hits

            #Loop through shuffles
            for iS in range(nShuffles):
                #Add hits to confusion matrix
                c_shf[iS] += loo_shf[iS]

        #Calculate decoding accuracy for this leave-1-out x-validation
        confusion_mat = confusion_mat/np.sum(confusion_mat,axis=1).reshape(-1,1)

        #Loop through shuffles
        for iS in range(nShuffles):
            #Calculate shuffled decoding accuracy for this leave-1-out shuffle
            c_shf[iS] = c_shf[iS]/np.sum(c_shf[iS],axis=1).reshape(-1,1)

        #Calculate z-score 
        m_shf, s_shf = np.mean(c_shf,axis=0), np.std(c_shf,axis=0)
        confusion_shf = m_shf
        confusion_z = (confusion_mat - m_shf)/s_shf

        #Get signficance of decoding 
        pvalues_loo = st.norm.sf(confusion_z)
        
        if plot_shuffle:
            #Plot shuffle distributions
            title = 'Leave-1-Out Cross-Validation'
            plot_decoding_shuffle(confusion_mat, c_shf, pvalues_loo, title,pdfdoc)
            
    elif method == 'kfold':       
        kfold_accuracies = []
        shf_accuracies = []
        kfold_zscores = []
        
        #Calculate decoding accuracy per kfold
        for iK,rTuple in enumerate(results):
            kfold_hits = rTuple[0] #size [nClasses x nClasses]
            kfold_shf = rTuple[1]  #size [nShuffles,nClasses x nClasses]

            #Normalize confusion matrix
            kfold_accuracies.append(kfold_hits/np.sum(kfold_hits,axis=1).reshape(-1,1))

            #Loop through shuffles and normalize
            c_shf = np.zeros((nShuffles,nClasses,nClasses))
            for iS in range(nShuffles):
                c_shf[iS] = kfold_shf[iS]/np.sum(kfold_shf[iS],axis=1).reshape(-1,1)

            #Calculate z-score for this kfold
            m_shf, s_shf = np.mean(c_shf,axis=0), np.std(c_shf,axis=0)
            shf_accuracies.append(m_shf)
            kfold_zscores.append((kfold_accuracies[iK] - m_shf)/s_shf)

            #Get signficance of decoding 
            pvalues_kfold = st.norm.sf(kfold_zscores[iK])
            
            if plot_shuffle:
                #Plot shuffle distributions
                title = 'Shuffle Distributions for kfold {}'.format(iK)
                plot_decoding_shuffle(kfold_accuracies[iK], c_shf, pvalues_kfold, title,pdfdoc)
        
        #Take average over kfolds
        confusion_mat = np.mean(kfold_accuracies,axis=0)
        confusion_shf = np.mean(shf_accuracies,axis=0)
        confusion_z = np.mean(kfold_zscores,axis=0)
        
    return confusion_mat, confusion_shf, confusion_z
    
def cross_validate(X,Y,Y_sort=None,method='L1O',nKfold=10,classifier='LDA',classifier_kws=None,clabels=None,shuffle=True,plot_shuffle=False,parallel=False,pdfdoc=None):
    ##===== Description =====##
    #The main difference between these 2 methods of cross-validation are that kfold approximates the decoding accuracy per kfold 
    #and then averages across folds to get an overall decoding accuracy. This is faster and a better approximation of the actual
    #decoding accuracy if you have enough data. By contrast, Leave-1-out creates just 1 overall decoding accuracy by creating 
    #a classifier for each subset of data - 1, and adding those results to a final confusion matrix to get an estimate of the 
    #decoding accuracy. While this is what you have to do in low-data situations, the classifiers are very similar and thus 
    #share a lot of variance. Regardless, I've written both ways of calculating the decoding accuracy below. 
    
    if Y_sort is None:
        Y_sort = Y
        
    #Leave-1(per group)-out
    if method == 'L1O':
        _,nTrials_class = np.unique(Y,return_counts=True)
        k_fold = StratifiedKFold(n_splits=nTrials_class[0])
    #Or k-fold
    elif method == 'kfold':
        k_fold = StratifiedKFold(n_splits=nKfold)
            
    #Multi-processing module is weird and might hang, especially with jupyter; try without first
    if parallel:
        pool = mp.Pool(processes=nProcesses)
        processes = []
    results = []
    
    ##===== Loop over cross-validation =====##
    for iK, (train_index, test_index) in enumerate(k_fold.split(X,Y_sort)):
#         print(np.unique(Y[train_index],return_counts=True))
#         print(np.unique(Y_sort[train_index],return_counts=True))
#         pdb.set_trace()
        if parallel:
            processes.append(pool.apply_async(decode_labels,args=(X,Y,train_index,test_index,classifier,clabels,X_test,shuffle,classifier_kws)))
        else:
            tmp = decode_labels(X,Y,train_index,test_index,classifier,clabels,None,None,shuffle,classifier_kws)
            results.append(tmp)

    #Extract results from parallel kfold processing
    if parallel:
        results = [p.get() for p in processes]
        pool.close()
        
    ##===== Calculate decoding accuracy =====##
    confusion_mat, confusion_shf, confusion_z = calculate_accuracy(results,method,plot_shuffle,pdfdoc)
    
    return confusion_mat, confusion_shf, confusion_z


##==============================##
##===== Plotting Functions =====##

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    

def plot_decoding_shuffle(decoding_accuracy, shuffles, pvalues,title=None,pdfdoc=None):
    
    nClasses = decoding_accuracy.shape[-1]
    ## Plot shuffle distributions ##
    fig,axes = plt.subplots(1,nClasses,figsize=(18,6))
    plt.suptitle(title,y=1.01)

    #Plot the shuffle distribution with the mean decoding performance for that class
    for i in range(nClasses):
        ax = axes[i]
        sns.distplot(shuffles[:,i,i],color=cc[i],ax=ax)
        if pvalues[i,i] < 0.01:
            ax.set_title('element [{},{}], pval: {:.1e}'.format(i,i,pvalues[i,i]))
        else:
            ax.set_title('element [{},{}], pval: {:.2f}'.format(i,i,pvalues[i,i]))

        ax.vlines(decoding_accuracy[i,i], *ax.get_ylim(),LineWidth=2.5,label='Data: {:.2f}'.format(decoding_accuracy[i,i]))
        ax.vlines(np.mean(shuffles,axis=0)[i,i], *ax.get_ylim(),LineWidth=2.5,LineStyle = '--',label='Shuffle: {:.2f}'.format(np.mean(shuffles,axis=0)[i,i]))
        ax.set_xlim(xmin=0)
        ax.legend()

    if pdfdoc is not None:
        pdfdoc.savefig(fig)
        plt.close(fig)

def plot_confusion_matrices(confusion_mat, confusion_z, plot_titles=None,class_labels=None,title=None, pdfdoc=None):

    nPlots = confusion_mat.shape[0]
    nClasses = confusion_mat.shape[-1]
    
    ##===== Plotting =====##
    fig,axes = plt.subplots(1,5,figsize=(16.25,4),gridspec_kw={'wspace': 0.25,'width_ratios':(4,4,4,4,0.25)})#,
    plt.suptitle(title,y=0.95)

    for iPlot in range(nPlots):
        ax = axes[iPlot]
        pvalues = st.norm.sf(confusion_z[iPlot])
        
        #Plot actual decoding performance
#         if iPlot == 4 :
        sns.heatmap(confusion_mat[iPlot],annot=True,fmt='.2f',annot_kws={'fontsize': 16},cbar=True,cbar_ax=axes[-1],square=True,cbar_kws={'shrink': 0.25,'aspect':1},vmin=np.percentile(confusion_mat[:],5),vmax=np.percentile(confusion_mat[:],95),ax=ax)
        ax.set_title('Image {}'.format(plot_titles[iPlot]),fontsize=14)
    
        for i in range(nClasses):
            for j in range(nClasses):
                if (pvalues[i,j] < 0.05):
                    ax.text(j+0.75,i+0.25,'*',color='g',fontsize=20,fontweight='bold')
        #Labels
        if class_labels is not None:
            if iPlot == 0:
                ax.set_ylabel('Actual Stimulus Block',fontsize=12)
                ax.set_yticklabels(class_labels, rotation=90,va="center",fontsize=12)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel('Decoded Stimulus Block',fontsize=12)
            ax.set_xticklabels(class_labels,va="center",fontsize=12) 
            
    if pdfdoc is not None:
        pdfdoc.savefig(fig)
        plt.close(fig)
    
def plot_decoding_accuracy(confusion_mat,confusion_z,ax=None,block='randomized_ctrl',class_labels=None,title=None,annot=True,clims=None):
    #Plot decoding performance
    if ax is None:
        fig,ax = plt.subplots(figsize=(6,6))#,
        
    if title is not None:
        ax.set_title(title,fontsize=14)

    pvalues = st.norm.sf(confusion_z)
    
    if clims is None:
        clims = [np.percentile(confusion_mat,1) % 0.05, np.percentile(confusion_mat,99) - np.percentile(confusion_mat,99) % 0.05]
    #Plot actual decoding performance
    sns.heatmap(confusion_mat,annot=annot,fmt='.2f',annot_kws={'fontsize': 16},cbar=True,square=True,vmin=clims[0],vmax=clims[1],cbar_kws={'shrink': 0.5},ax=ax)

    nClasses = confusion_mat.shape[0] 
    for i in range(nClasses):
        for j in range(nClasses):
            if (pvalues[i,j] < 0.05):
                ax.text(j+0.65,i+0.35,'*',color='g',fontsize=20,fontweight='bold')
    #Labels
    ax.set_ylabel('Actual Mouse',fontsize=14)
    ax.set_xlabel('Decoded Mouse',fontsize=14)
    if class_labels is not None:
        ax.set_yticklabels(class_labels,va="center",rotation=45,fontsize=14)
        ax.set_xticklabels(class_labels,va="center",rotation=45,fontsize=14) 