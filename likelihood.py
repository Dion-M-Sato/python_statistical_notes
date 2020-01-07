import pandas as pd
import numpy as np

#Basical Idea
#MLE is solved by EM Algorithm which "calculates the expected value of the log-likelihood given the data and prior distribution of the parameters, then calculates the maximum value of this expected value of the log-likelihood function given those parameters. "(https://towardsdatascience.com/maximum-likelihood-estimation-how-it-works-and-implementing-in-python-b0eb2efb360f) 

def expectation_max(data, max_iter=1000):
    data = pd.DataFrame(data)
    mu0 = data.mean()
    c0 = data.cov()

    for j in range(max_iter):
        w = []
        #perform the expetation values of the algorithm
        for i in data:
            wk = (5+len(data))/(5+np.dot(np.transpose(i-mu0), np.linalg.inv(c0)),(i-mu0)))
            w.append(wk)
            w = np.array(w)
        #perform the Mahalanobis distance of the algorithm
        mu = (np.dot(w, data))/(np.sum(w))

        c = 0
        for i in range (len(data)):
            c += w[i] * np.dot((data[i]-mu0),(np.transpose(data[i]-mu0)))
            cov = c/len(data)
            mu0 = mu
            c0 = cov
    return mu0, c0
# It returns estimated sample mean and covariance.
