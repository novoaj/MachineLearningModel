# project: p7
# submitter: jnovoa@wisc.edu
# partner: none
# hours: 10
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, cross_val_score

class UserPredictor:
    def __init__(self):      
        self.xcols = ["past_purchase_amt", "badge", "age", "seconds"]
        self.numericals = ["past_purchase_amt", "age", "seconds"]
        self.categoricals = ["badge"]       
        
    def fit(self, users_df, logs_df, y):        

        new_df = pd.merge(users_df, logs_df.groupby(by="user_id")["seconds"].sum(), how = "left", on = "user_id").fillna(0) # create new df with seconds column and handle NaN by replacing with 0
        # new_df["seconds"].replace(to_replace = 0, value = new_df["seconds"].mean(), inplace = True) # replace 0s with mean of column (this hurts our predictions)
         
        custom_trans = make_column_transformer(
            # (StandardScaler(), self.numericals),
            (OneHotEncoder(), self.categoricals),
            (PolynomialFeatures(degree=2), self.numericals),
            
        )
        
        self.model = Pipeline([
            ("custom", custom_trans),
            ("std", StandardScaler()),
            ("lr", LogisticRegression(fit_intercept = False)),
        ])

        self.model.fit(new_df[self.xcols], y["y"]) # fit using xcols to predict y
    
    def predict(self, users_df,logs_df):
        new_df = pd.merge(users_df, logs_df.groupby(by="user_id")["seconds"].sum(), how = "left", on = "user_id").fillna(0)

        return self.model.predict(new_df[self.xcols]) # prediction of y
