'''
Created on Nov 29, 2014

@author: James Thornton
'''

from sklearn import preprocessing, datasets, svm, feature_selection
import matplotlib.pyplot as plt
import scipy.io

class prepro:
    '''
    classdocs
    '''
    
    def __init__(self, datafile):
        self.file = scipy.io.loadmat(datafile)
        
    def process(self):
        for l in self.file:
            print l
            print self.file[l]
        
        X = preprocessing.scale(self.file['data'])
        y = range(len(X))
        print X.shape
        print len(X[0])
        
        svc = svm.SVC(kernel="linear",C=1)
        fe_sel = 1
        print self.file['data']
        rfe = feature_selection.RFE(estimator=svc, n_features_to_select=fe_sel, step=1)
        
        rfe.fit(X, y)
        ranking = rfe.ranking_.reshape((1,rfe.ranking_.shape[0]))
        print ranking.shape
        
        ''''''''''''''''''
        
        for i in X:
            plt.plot(range(len(i)),i)
        
        plt.matshow(ranking)
        plt.colorbar()
        plt.show()