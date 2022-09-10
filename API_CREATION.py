#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as s
from sklearn.metrics import classification_report


# In[2]:


class gaussian_naive:
    """
    features: all the features in data,excluding labels
    labels : a series consisting of binary labels
    data_splitting:  a tuple consisting of data split ratio
    apply_pca : boolean_value specifying whether pca apply or not
    n_components: number of eigne vector that having non zero to keep
    
    """
    
    
    def __init__(xerox_copy,features,labels,data_spilitting,apply_pca,n_components):
        xerox_copy.binary_labels = np.array(labels).reshape(labels.shape[0],1)
        xerox_copy.split_ratio = data_splitting
        xerox_copy.n_principal_component =n_components
        xerox_copy.unique_labels =list(labels.unique())
        if apply_pca == True:
            xerox_copy.X_new =xerox_copy.apply_dim_reduction(features,xerox_copy.n_principal_component)
    
    def apply_dim_reduction(xerox_copy,data,n_components):
        X =np.array(data)
        mu_hat = np.mean(X,axis=0)
        X_dash =X -mu_hat
        sigma_hat =(1/data.shape[0] )*np.matmul(x_dash.T,X_dash)
        sigma_hat_decompose=np.linalg.svd(sigma_hat)
        Q =sigma_hat_decompose[0]
        Q_tilda =Q[:,0:n_components]
        X_new=np.matmul(X_dash,Q_tilda)
        
        return X_new
    
    def data_splitting_ratio(xerox_copy):
        new_data=pd.DataFrame(data=xerox_copy.X_new)
        new_data['labels']=xerox_copy.binary_labels
        X_train_len =int(xerox_copy.split_ratio[0]*new_data.shape[0])
        neg_training_data =new_data[new_data['labels']==xerox_copy.unique_labels[0]].iloc[:,0:X_train_len//2]
        pos_training_data=new_data[new_data['labels']==xerox_copy.unique_labels[1]].iloc[:,0:X_train_len//2]
        train_data= pd.concat([neg_training_data,pos_training_data])
        cv_data_len=int(xerox_copy.split_ratio[1]*new_data.shape[0])
        neg_remaining_data=new_data[new_data['labels']==xerox_copy.unique_labels[0]].iloc[:,X_train_len: ]
        pos_remaining_data=new_data[new_data['labels']==xerox_copy.unique_labels[1]].iloc[:,X_train_len: ]
        remaining_data=pd.concat([neg_remaining,pos_remaining_data])
        cv_data=remaining_data.iloc[:,0:cv_data_len]
        test_data=remaining_data.iloc[:,cv_data_len: ]
        
        return train_data,cv_data,test_data
    def train_gaussian(xerox_copy,data):
        mu_hat_neg=np.array(data[data['labels']==xerox_copy.unique_labels[0]].iloc[:,0:data.shape[0]//2]).mean()
        sigma_hat_neg=np.array(data[data['labels']==xerox_copy.unique_labels[0]].iloc[:,0:data.shape[0]//2]).cov()
        mu_hat_pos=np.array(data[data['labels']==xerox_copy.unique_labels[1]].iloc[:,0:data.shape[0]//2]).mean()
        sigma_hat_pos=np.array(data[data['labels']==xerox_copy.unique_labels[1]].iloc[:,0:data.shape[0]//2]).cov()
        #now we are doing monkey patching
        xerox_copy.pos_likelihood=(mu_hat_pos,sigma_hat_pos)
        xerox_copy.neg_likelihood=(mu_hat_neg,sigma_hat_neg)
        
        
    def evaluate(xerox_copy,data):
        inputs =np.array(data)
        posterior_pos =s.multivariate_normal.pdf(inputs,xerox_copy.pos_likelihood)
        posterior_neg=s.multivariate_normal.pdf(inputs,xerox_copy.neg_likelihood)
        boolean_mask=posterior_pos > posterior_neg
        predicted_category =pd.Series(boolean_mask)
        predicted_category.replace(to_replace=[False,True],value=[xerox_copy.unique_labels[0],xerox_copy.unique_labels[1]])
        predicted_results=np.array(predicted_category)
        actual_results =np.array(data['labels'])
        print(classification_report(actual_results,predicted_results,target_names=xerox_copy.unique_labels))
        
        
        
        
        


# In[3]:


if __name__ =="__main__":
    print("module is created")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




