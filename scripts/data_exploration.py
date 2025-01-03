import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

## Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataExploration:
    """ This Class is Used for Data Exploration"""

    def __init__(self,data):
        self.data = data


    def test_data_distribution_1(self):

        """ Plots graphs to analyse Test data distribution"""
        
        logging.info("Ploting Distribution graphs for Test Data ... ")

        fig, ax = plt.subplots(2,2, figsize=(12,8))
        sns.countplot(data=self.data, x='StoreType', ax=ax[0][0])
        sns.countplot(data=self.data, x='Assortment', ax=ax[0][1])
        sns.countplot(data=self.data, x='StateHoliday', ax=ax[1][0])
        sns.countplot(data=self.data, x='SchoolHoliday', ax=ax[1][1])
        plt.tight_layout()

        logging.info("Distribution graph Ploted!!")

    def test_data_distribution_2(self):

       """ Plots graphs to analyse Test data distribution"""
        
       logging.info("Ploting Distribution graphs for Test Data ... ")

       fig, ax = plt.subplots(2,2, figsize=(12,8))
       sns.countplot(data=self.data, x='Promo2', ax=ax[0][0])
       sns.countplot(data=self.data, x='Promo', ax=ax[0][1])
       sns.countplot(data=self.data, x='Open', ax=ax[1][0])
       plt.tight_layout()

       logging.info("Distribution graph Ploted!!")

    def train_data_distribution_1(self):

       """ Plots graphs to analyse Train data distribution"""
        
       logging.info("Ploting Distribution graphs for Training Data ... ")

       fig, ax = plt.subplots(2,2, figsize=(12,8))
       sns.countplot(data=self.data, x='StoreType', ax=ax[0][0])
       sns.countplot(data=self.data, x='Assortment', ax=ax[0][1])
       sns.countplot(data=self.data, x='StateHoliday', ax=ax[1][0])
       sns.countplot(data=self.data, x='SchoolHoliday', ax=ax[1][1])
       plt.tight_layout()

       logging.info("Distribution graph Ploted!!")

    def train_data_distribution_2(self):

       """ Plots graphs to analyse Train data distribution"""
        
       logging.info("Ploting Distribution graphs for Training Data ... ")

       fig, ax = plt.subplots(2,3, figsize=(12,8))
       sns.countplot(data=self.data, x='Promo2', ax=ax[0][0])
       sns.countplot(data=self.data, x='Promo', ax=ax[0][1])
       sns.countplot(data=self.data, x='Open', ax=ax[0][2])
       sns.histplot(data=self.data, x='Sales', bins=10, kde=True, ax=ax[1][0])
       sns.histplot(data=self.data, x='Customers', bins=10, kde=True, ax=ax[1][1])
       plt.tight_layout()

       logging.info("Distribution graph Ploted!!")

    def outliers(self):
        """ Checks for outliers in given columns"""

        logging.info("Ploting Box Plots ... ")
        fig, ax = plt.subplots(1,2, figsize=(12,6))
        sns.boxplot(data=self.data, x=self.data['Customers'],ax=ax[0])
        sns.boxplot(data=self.data, x=self.data['Sales'],ax=ax[1])
        plt.tight_layout()
        logging.info("Box Plot plotted!!")

    def handle_outliers(self):
        """ Imputes outliers appropriately"""
        columns = ['Sales', 'Customers']
        for col in columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5*IQR
            upper_bound = Q3 + 1.5*IQR

            self.data[col] = np.where(((self.data[col] <= lower_bound) | (self.data[col] >= upper_bound)), self.data[col].mean(), self.data[col])

        return self.data
    
    
    def bivariate_analysis_1(self):
        """ Checks for outliers in given columns"""

        logging.info("Plotting Bar Charts for bivariate analysis ... ")
        fig, ax = plt.subplots(2,2, figsize=(12,8))
        sns.barplot(data=self.data, y='Sales', x='Promo',ax=ax[0][0])
        sns.barplot(data=self.data, y='Sales', x='Promo2',ax=ax[0][1])
        sns.barplot(data=self.data, y='Sales', x='Assortment',ax=ax[1][0])
        sns.barplot(data=self.data, y='Sales', x='StateHoliday',ax=ax[1][1])
        plt.tight_layout()
        logging.info("Chart Plotted!!")
    
    def bivariate_analysis_2(self):
        """ Checks for outliers in given columns"""

        logging.info("Plotting Bar Charts for bivariate analysis ... ")
        fig, ax = plt.subplots(2,2, figsize=(12,8))
        sns.barplot(data=self.data, y='Sales', x='SchoolHoliday',ax=ax[0][0])
        sns.barplot(data=self.data, y='Sales', x='DayOfWeek',ax=ax[0][1])
        sns.barplot(data=self.data, y='Sales', x='StoreType',ax=ax[1][0])
        plt.tight_layout()
        logging.info("Chart Plotted!!")

    