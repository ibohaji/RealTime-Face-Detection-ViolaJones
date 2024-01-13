import numpy as np
import math 
import time 


class DecissionStump:


     def __init__(self,idx,threshold = None,polarity=1,error=2,margin=0):
        self.threshold = threshold 
        self.polarity = polarity
        self.error = error 
        self.margin = margin
        self.idx = idx 

     def predict(self,X):
        
        observations = X[self.idx]
        yhat = np.ones(X.shape)

        if self.polarity == 1: 
            yhat = np.where(observations>self.threshold,1,0)
        else: 
            yhat = np.where(observations<self.threshold,1,0)

        return yhat 

     def train(self,X,y,w):

        sorted_idx = np.argsort(X)
        original_idx = np.argsort(sorted_idx) 

        #Sorting the values in ascending order of features
        X = X[sorted_idx] 
        y = y[sorted_idx]
        w = w[sorted_idx]
        N = len(X) 

        margin = 0 
        error = 2 
        p = 1
      
        for i in range(0,N):
            threshold,margin = ((X[i] + X[i+1])/2,X[i+1]-X[i]) if i<N-1 else (X[i] - 1,0)

            w_p_h = np.sum(w[(X>threshold) & (y == 1)])
            w_n_h = np.sum(w[(X>threshold) & (y == 0)])

            w_p_l = np.sum(w[(X<threshold) & (y == 1)])
            w_n_l = np.sum(w[(X<threshold) & (y == 0)])


            error_pos = w_p_l + w_n_h 
            error_neg = w_n_l + w_p_h 

            if error_pos < error_neg:
                error = error_pos
                p = 1
            else:
                error = error_neg
                p = -1

            if (self.error > error and error!=0) or ( (error == self.error) and (margin > self.margin) and (error!=0) ):
                self.error = error 
                self.margin = margin 
                self.threshold = threshold
                self.polarity = p
        
        
        return self.error,self.margin
    



         
        
def best_stump(X,y,w): 

        d = X.shape[0]
        best_error = 2 
        widest_margin = 0 
        best_clf = None 
        best_feature = 0 

        for f in range(0,d):
            
            clf = DecissionStump(idx=f)
            error,margin = clf.train(X[f],y,w)
          #  print("error in line 89 is : {}".format(error))
            if(error < best_error) or ((error == best_error) & (widest_margin < margin)):
                best_error = error 
                widest_margin = margin
                best_clf = clf
                best_feature = f

        print("found threshold of iteration {}: {}".format(best_feature,best_clf.threshold))
        print("Feature used: {}".format(best_feature))
        return best_clf 

class AdaBoost:
    def __init__(self,T):
        self.T = T 

    def run(self,X,labels): 

        N,M = X.shape 
        N = len(labels)
        self.ht = np.empty(self.T,dtype=object) # For the weak learners
        self.alphas = np.empty(self.T-1) 
        print("Shape of features: {} x {}".format(N,M))

        #len of negative and positive examples
        l = np.sum(labels==0)
        m = np.sum(labels==1)

        w_p = 1/(2*m) #Weight of each pos 
        w_n = 1/(2*l) #Weight of each pos 
        wt =np.where(labels==1,w_p,w_n) #vector of weight according to lable 

        for t in range(self.T-1):
            print("iteration nr: {}".format(t))
            wt/= np.sum(wt)  # Normalize the weights


            clf = best_stump(X,labels,wt) 
            error = clf.error 
            yhat = clf.predict(X)

            print("Clf's threshold: {}".format(clf.threshold))
            print("Clf's error: {}".format(clf.error))
            print("Clf's feature index : {}".format(clf.idx))
            alpha = np.log((1-error)/error)
            
            if(error == 0):
                break

            self.ht[t] = clf 
            self.alphas[t] = alpha 

            print("Sum of the new weights before: {}".format(np.sum(wt)))
            print("Clf error: {}".format(clf.error))
            print("wt.shape: {}".format(wt.shape))
            print("pred.shape: {}".format(clf.predict(X).shape))
            print("labels.shape: {}".format(labels.shape))
            wt*= np.where(clf.predict(X)==labels, clf.error/(1-clf.error),1)
            print("Sum of the new weights after: {}".format(np.sum(wt)))

    def predict(self,X): 
        ys = []
        for alpha,classifier in zip(self.alphas,self.ht):
            ys.append(alpha*classifier.predict(X))
        
        return (np.sum(ys,axis=0) >= 0.5*sum(self.alphas))

    def accuracy(self,predicted,true):
        print("Predicted")
        correct = np.sum(predicted == true)
        total_samples = len(true)
        accuracy = correct / total_samples
        return accuracy

                

        



                


            



