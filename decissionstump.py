import numpy as np


class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.threshold = None
        self.error = float('inf')

    def train(self, features, labels):
        # Sort the data by feature values and get the sorted labels
        sorted_indices = features.argsort()
        sorted_features = features[sorted_indices]
        sorted_labels = labels[sorted_indices]

        # Initialize weights
        n_samples = len(labels)
        weight_positive = np.sum(labels == 1) / n_samples
        weight_negative = np.sum(labels == 0) / n_samples

        # Initialize errors for positive and negative polarities
        error_positive = weight_positive * np.sum(sorted_labels == 0)  # All positive, initially
        error_negative = weight_negative * np.sum(sorted_labels == 1)  # All negative, initially

        for i in range(1, n_samples):
            # Update errors based on the current threshold
            if sorted_labels[i-1] == 1:
                error_positive += weight_positive
                error_negative -= weight_negative
            else:
                error_positive -= weight_positive
                error_negative += weight_negative

            # Check if this threshold results in lower error
            if i == n_samples - 1 or sorted_features[i] != sorted_features[i-1]:
                if error_positive < self.error or error_negative < self.error:
                    self.error = min(error_positive, error_negative)
                    self.threshold = sorted_features[i]
                    self.polarity = 1 if error_positive < error_negative else -1
        
        self.margin = np.abs(features - self.threshold)

        return self.threshold,self.polarity,self.error,self.margin 


    def predict(self, X):
        # Make predictions based on the trained threshold and polarity
        predictions = np.ones(len(X))
        if self.polarity == 1:
            predictions[X < self.threshold] = 0
        else:
            predictions[X >= self.threshold] = 0
        return predictions


class AdaBoost:
    def __init__(T):
        self.T = T 

    def best_stump(self,feature,labels):
        best_error = 2
        best_margin = 0
        best_clf = None 

        for i in range(0,feature):
            clf = DecisionStump()
            threshold,polarity,error,margin = clf.train(feature,labels)
            if error<best_error:
                best_error = error 
                best_clf = clf
            elif margin>best_margin:
                best_margin = margin 
                best_clf = clf 
    
  #  def train(self,data):


                

        



                


            



