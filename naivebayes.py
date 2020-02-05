
"""
This is the implementation of NaiveBayes Classification Algorithm

"""
import numpy as np

class NaiveBayes:
    #initialization of prameters
    def fit(self,X,y):
        n_samples,n_features=X.shape
        self._classes=np.unique(y) #finds unique elements of an array
        n_classes=len(self._classes)#number of unique classes
        #init mean,varience,priors
        self._mean=np.zeros((n_samples,n_features),dtype=np.float64)
        self._var=np.zeros((n_samples,n_features),dtype=np.float64)
        self._priors=np.zeros(n_classes,dtype=np.float64)
        
        for c in self._classes:
            X_c=X[c==y] #we only want the samples that has c as label
            self._mean[c,:]=X_c.mean(axis=0)
            self._var[c,:]=X_c.var(axis=0)
            self._priors[c]=X_c.shape[0]/float(n_samples)  #prior probability of class c
            
    def predict(self,X):  
        y_pred=[self.pred(x) for x in X ]
        return y_pred
    
    #Prediction Method
    def pred(self,x):
        posteriors=[]
        for i,c in enumerate (self._classes):
            prior= np.log(self._priors[i])
            class_conditional=np.sum(np.log(self.pdf(i , x)))
            posterior=prior+class_conditional
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]
    
   #Class conditional probability or probability density function
    def pdf(self,class_i,x):
        mean=self._mean[class_i]
        var=self._var[class_i]
        #pxy is the probability of x in class y
        n=np.exp(-(x-mean)**2/(2*var))
        d=np.sqrt(2* np.pi *var)
        pxy=n/d
        return pxy
   