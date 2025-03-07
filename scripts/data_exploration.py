import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

## Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress INFO-level logs from the library
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class DataExploration:
    """ This Class is Used for Data Exploration"""

    def __init__(self,data):
        self.data = data


    def test_data_distribution_1(self):

        """ Plots graphs to analyse Test data distribution"""
        
        logging.info("Ploting Distribution graphs for Test Data ... ")

        fig, ax = plt.subplots(2,2, figsize=(12,8))
        sns.countplot(data=self.data, x='StoreType', ax=ax[0][0])
        ax[0][0].set_title('Store Type Distribution', fontsize=12, fontweight='bold') 
        ax[0][0].set_xlabel('Store Type', fontsize=10, fontweight='bold')           
        ax[0][0].set_ylabel('Count', fontsize=10, fontweight='bold')                

        sns.countplot(data=self.data, x='Assortment', ax=ax[0][1])
        ax[0][1].set_title('Assortment Distribution', fontsize=12, fontweight='bold')  
        ax[0][1].set_xlabel('Assortment', fontsize=10, fontweight='bold')             
        ax[0][1].set_ylabel('Count', fontsize=10, fontweight='bold')                  

        sns.countplot(data=self.data, x='StateHoliday', ax=ax[1][0])
        ax[1][0].set_title('State Holiday Distribution', fontsize=12, fontweight='bold')  
        ax[1][0].set_xlabel('State Holiday', fontsize=10, fontweight='bold')            
        ax[1][0].set_ylabel('Count', fontsize=10, fontweight='bold')                   

        sns.countplot(data=self.data, x='SchoolHoliday', ax=ax[1][1])
        ax[1][1].set_title('School Holiday Distribution', fontsize=12, fontweight='bold')
        ax[1][1].set_xlabel('School Holiday', fontsize=10, fontweight='bold')            
        ax[1][1].set_ylabel('Count', fontsize=10, fontweight='bold')    

        plt.tight_layout()
        plt.show() 

        logging.info("Distribution graph Ploted!!")

    def test_data_distribution_2(self):

        """ Plots graphs to analyse Test data distribution"""
            
        logging.info("Ploting Distribution graphs for Test Data ... ")

        fig, ax = plt.subplots(2,2, figsize=(12,8))
        # Plot 1: Promo2 Distribution
        sns.countplot(data=self.data, x='Promo2', ax=ax[0][0])
        ax[0][0].set_title('Promo2 Distribution', fontsize=14, fontweight='bold')  # Title
        ax[0][0].set_xlabel('Promo2', fontsize=12, fontweight='bold')             # X-axis label
        ax[0][0].set_ylabel('Count', fontsize=12, fontweight='bold')              # Y-axis label

        # Plot 2: Promo Distribution
        sns.countplot(data=self.data, x='Promo', ax=ax[0][1])
        ax[0][1].set_title('Promo Distribution', fontsize=14, fontweight='bold')  # Title
        ax[0][1].set_xlabel('Promo', fontsize=12, fontweight='bold')              # X-axis label
        ax[0][1].set_ylabel('Count', fontsize=12, fontweight='bold')              # Y-axis label

        # Plot 3: Open Distribution
        sns.countplot(data=self.data, x='Open', ax=ax[1][0])
        ax[1][0].set_title('Open Distribution', fontsize=14, fontweight='bold')   # Title
        ax[1][0].set_xlabel('Open', fontsize=12, fontweight='bold')               # X-axis label
        ax[1][0].set_ylabel('Count', fontsize=12, fontweight='bold')              # Y-axis label

        # Remove the last axis (bottom-right subplot)
        ax[1][1].axis('off')
        plt.tight_layout()

        logging.info("Distribution graph Ploted!!")

    def train_data_distribution_1(self):

        """ Plots graphs to analyse Train data distribution"""
        
        logging.info("Ploting Distribution graphs for Training Data ... ")

        fig, ax = plt.subplots(2,2, figsize=(12,8))
        # Plot 1: StoreType Distribution
        sns.countplot(data=self.data, x='StoreType', ax=ax[0][0])
        ax[0][0].set_title('Store Type Distribution', fontsize=14, fontweight='bold')  # Title
        ax[0][0].set_xlabel('Store Type', fontsize=12, fontweight='bold')             # X-axis label
        ax[0][0].set_ylabel('Count', fontsize=12, fontweight='bold')                 # Y-axis label

        # Plot 2: Assortment Distribution
        sns.countplot(data=self.data, x='Assortment', ax=ax[0][1])
        ax[0][1].set_title('Assortment Distribution', fontsize=14, fontweight='bold')  # Title
        ax[0][1].set_xlabel('Assortment', fontsize=12, fontweight='bold')             # X-axis label
        ax[0][1].set_ylabel('Count', fontsize=12, fontweight='bold')                  # Y-axis label

        # Plot 3: StateHoliday Distribution
        sns.countplot(data=self.data, x='StateHoliday', ax=ax[1][0])
        ax[1][0].set_title('State Holiday Distribution', fontsize=14, fontweight='bold')  # Title
        ax[1][0].set_xlabel('State Holiday', fontsize=12, fontweight='bold')             # X-axis label
        ax[1][0].set_ylabel('Count', fontsize=12, fontweight='bold')                    # Y-axis label

        # Plot 4: SchoolHoliday Distribution
        sns.countplot(data=self.data, x='SchoolHoliday', ax=ax[1][1])
        ax[1][1].set_title('School Holiday Distribution', fontsize=14, fontweight='bold')  # Title
        ax[1][1].set_xlabel('School Holiday', fontsize=12, fontweight='bold')             # X-axis label
        ax[1][1].set_ylabel('Count', fontsize=12, fontweight='bold')    
        plt.tight_layout()

        logging.info("Distribution graph Ploted!!")

    def train_data_distribution_2(self):

        """ Plots graphs to analyse Train data distribution"""
        
        logging.info("Ploting Distribution graphs for Training Data ... ")

        fig, ax = plt.subplots(2,3, figsize=(12,8))
        # Plot 1: Promo2 Distribution
        sns.countplot(data=self.data, x='Promo2', ax=ax[0][0])
        ax[0][0].set_title('Promo2 Distribution', fontsize=14, fontweight='bold')  # Title
        ax[0][0].set_xlabel('Promo2', fontsize=12, fontweight='bold')             # X-axis label
        ax[0][0].set_ylabel('Count', fontsize=12, fontweight='bold')              # Y-axis label

        # Plot 2: Promo Distribution
        sns.countplot(data=self.data, x='Promo', ax=ax[0][1])
        ax[0][1].set_title('Promo Distribution', fontsize=14, fontweight='bold')  # Title
        ax[0][1].set_xlabel('Promo', fontsize=12, fontweight='bold')             # X-axis label
        ax[0][1].set_ylabel('Count', fontsize=12, fontweight='bold')             # Y-axis label

        # Plot 3: Open Distribution
        sns.countplot(data=self.data, x='Open', ax=ax[0][2])
        ax[0][2].set_title('Open Distribution', fontsize=14, fontweight='bold')  
        ax[0][2].set_xlabel('Open', fontsize=12, fontweight='bold')           
        ax[0][2].set_ylabel('Count', fontsize=12, fontweight='bold')            

        # Plot 4: Sales Distribution
        sns.histplot(data=self.data, x='Sales', bins=10, kde=True, ax=ax[1][0])
        ax[1][0].set_title('Sales Distribution', fontsize=14, fontweight='bold')  
        ax[1][0].set_xlabel('Sales', fontsize=12, fontweight='bold')             
        ax[1][0].set_ylabel('Frequency', fontsize=12, fontweight='bold')         

        # Plot 5: Customers Distribution
        sns.histplot(data=self.data, x='Customers', bins=10, kde=True, ax=ax[1][1])
        ax[1][1].set_title('Customers Distribution', fontsize=14, fontweight='bold')  
        ax[1][1].set_xlabel('Customers', fontsize=12, fontweight='bold')         
        ax[1][1].set_ylabel('Frequency', fontsize=12, fontweight='bold')             

        # Remove the last axis (bottom-right subplot)
        ax[1][2].axis('off')
        plt.tight_layout()

        logging.info("Distribution graph Ploted!!")

    def outliers(self):
        """ Checks for outliers in given columns"""

        logging.info("Ploting Box Plots ... ")
        fig, ax = plt.subplots(1,2, figsize=(12,6))
        # Plot 1: Customers Boxplot
        sns.boxplot(data=self.data, x=self.data['Customers'], ax=ax[0])
        ax[0].set_title('Customers Boxplot', fontsize=14, fontweight='bold') 
        ax[0].set_xlabel('Customers', fontsize=12, fontweight='bold')        
        ax[0].set_ylabel('Values', fontsize=12, fontweight='bold')          

        # Plot 2: Sales Boxplot
        sns.boxplot(data=self.data, x=self.data['Sales'], ax=ax[1])
        ax[1].set_title('Sales Boxplot', fontsize=14, fontweight='bold')     
        ax[1].set_xlabel('Sales', fontsize=12, fontweight='bold')           
        ax[1].set_ylabel('Values', fontsize=12, fontweight='bold')    
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
        # Plot 1: Sales vs Promo
        sns.barplot(data=self.data, y='Sales', x='Promo', ax=ax[0][0])
        ax[0][0].set_title('Sales vs Promo', fontsize=14, fontweight='bold')  # Title
        ax[0][0].set_xlabel('Promo', fontsize=12, fontweight='bold')         # X-axis label
        ax[0][0].set_ylabel('Sales', fontsize=12, fontweight='bold')         # Y-axis label

        # Plot 2: Sales vs Promo2
        sns.barplot(data=self.data, y='Sales', x='Promo2', ax=ax[0][1])
        ax[0][1].set_title('Sales vs Promo2', fontsize=14, fontweight='bold')  # Title
        ax[0][1].set_xlabel('Promo2', fontsize=12, fontweight='bold')         # X-axis label
        ax[0][1].set_ylabel('Sales', fontsize=12, fontweight='bold')          # Y-axis label

        # Plot 3: Sales vs Assortment
        sns.barplot(data=self.data, y='Sales', x='Assortment', ax=ax[1][0])
        ax[1][0].set_title('Sales vs Assortment', fontsize=14, fontweight='bold')  # Title
        ax[1][0].set_xlabel('Assortment', fontsize=12, fontweight='bold')         # X-axis label
        ax[1][0].set_ylabel('Sales', fontsize=12, fontweight='bold')              # Y-axis label

        # Plot 4: Sales vs StateHoliday
        sns.barplot(data=self.data, y='Sales', x='StateHoliday', ax=ax[1][1])
        ax[1][1].set_title('Sales vs StateHoliday', fontsize=14, fontweight='bold')  # Title
        ax[1][1].set_xlabel('StateHoliday', fontsize=12, fontweight='bold')         # X-axis label
        ax[1][1].set_ylabel('Sales', fontsize=12, fontweight='bold')    
        plt.tight_layout()
        logging.info("Chart Plotted!!")
    
    def bivariate_analysis_2(self):
        """ Checks for outliers in given columns"""

        logging.info("Plotting Bar Charts for bivariate analysis ... ")
        fig, ax = plt.subplots(2,2, figsize=(12,8))
        # Plot 1: Sales vs SchoolHoliday
        sns.barplot(data=self.data, y='Sales', x='SchoolHoliday', ax=ax[0][0])
        ax[0][0].set_title('Sales vs SchoolHoliday', fontsize=14, fontweight='bold')  # Title
        ax[0][0].set_xlabel('SchoolHoliday', fontsize=12, fontweight='bold')         # X-axis label
        ax[0][0].set_ylabel('Sales', fontsize=12, fontweight='bold')                 # Y-axis label

        # Plot 2: Sales vs DayOfWeek
        sns.barplot(data=self.data, y='Sales', x='DayOfWeek', ax=ax[0][1])
        ax[0][1].set_title('Sales vs DayOfWeek', fontsize=14, fontweight='bold')  # Title
        ax[0][1].set_xlabel('DayOfWeek', fontsize=12, fontweight='bold')         # X-axis label
        ax[0][1].set_ylabel('Sales', fontsize=12, fontweight='bold')            # Y-axis label

        # Plot 3: Sales vs StoreType
        sns.barplot(data=self.data, y='Sales', x='StoreType', ax=ax[1][0])
        ax[1][0].set_title('Sales vs StoreType', fontsize=14, fontweight='bold')  # Title
        ax[1][0].set_xlabel('StoreType', fontsize=12, fontweight='bold')         # X-axis label
        ax[1][0].set_ylabel('Sales', fontsize=12, fontweight='bold')             # Y-axis label

        # Remove the last axis (bottom-right subplot)
        ax[1][1].axis('off')
        plt.tight_layout()
        logging.info("Chart Plotted!!")
    
    def bivariate_analysis_3(self):

        logging.info("Plotting Bar Charts for bivariate analysis ... ")
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=self.data, y='Sales', x='Customers')
        plt.title("Relationship Between Customers and Sales", fontsize=14, fontweight='bold')
        plt.xlabel("Number of Customers", fontsize=12, fontweight='bold')
        plt.ylabel("Sales", fontsize=12, fontweight='bold')
        plt.show()

        logging.info("Chart Plotted!!")

    def correlation_plot(self):
        
        logging.info("Plotting heatmap chart... ")
        plt.figure(figsize=(12, 8))  # Set the figure size
        sns.heatmap(self.data[['Sales', 'Customers']].corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Heatmap: Sales vs Customers', fontsize=14, fontweight='bold')
        plt.xlabel('Features', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()
        logging.info("Heatmap plotted!")

    def holiday_plot(self):


        # Define Holiday periods
        holiday_date = self.data[self.data['StateHoliday'] == 'a']['Date'].mode()[0]

        before_holiday_date = holiday_date - pd.Timedelta(days=7)
        after_holiday_date = holiday_date + pd.Timedelta(days=7)

        # Update the 'holidays' column based on conditions
        self.data['holidays'] = 'None'
        self.data.loc[(self.data['Date'] >= before_holiday_date) & (self.data['Date'] < holiday_date), 'holidays'] = 'Before'
        self.data.loc[self.data['Date'] == holiday_date, 'holidays'] = 'During'
        self.data.loc[(self.data['Date'] > holiday_date) & (self.data['Date'] <= after_holiday_date), 'holidays'] = 'After'

        logging.info("Plotting Bar chart... ")

        # Create the bar plot
        plt.figure(figsize=(12, 6))  # Set the figure size
        sns.barplot(data=self.data, x='holidays', y='Sales', palette='viridis')

        # Add title and labels
        plt.title("Sales Distribution During, Before, and After Public Holidays", fontsize=14, fontweight='bold')
        plt.xlabel('Holiday Period', fontsize=12, fontweight='bold')
        plt.ylabel('Sales', fontsize=12, fontweight='bold')

        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right') 

        # Show the plot
        plt.tight_layout()
        plt.show()

        logging.info("Bar Chart Plotted!!")

    def easter_plot(self):

        # Define Holiday periods
        holiday_date = self.data[self.data['StateHoliday'] == 'b']['Date'].mode()[0]

        before_holiday_date = holiday_date - pd.Timedelta(days=7)
        after_holiday_date = holiday_date + pd.Timedelta(days=7)

        # Update the 'holidays' column based on conditions
        self.data['holidays'] = 'None'
        self.data.loc[(self.data['Date'] >= before_holiday_date) & (self.data['Date'] < holiday_date), 'holidays'] = 'Before'
        self.data.loc[self.data['Date'] == holiday_date, 'holidays'] = 'During'
        self.data.loc[(self.data['Date'] > holiday_date) & (self.data['Date'] <= after_holiday_date), 'holidays'] = 'After'

        logging.info("Plotting Bar chart... ")

        # Create the bar plot
        plt.figure(figsize=(12, 6))  # Set the figure size
        sns.barplot(data=self.data, x='holidays', y='Sales', palette='viridis')

        # Add title and labels
        plt.title("Sales Distribution During, Before, and After Easter", fontsize=14, fontweight='bold')
        plt.xlabel('Holiday Period', fontsize=12, fontweight='bold')
        plt.ylabel('Sales', fontsize=12, fontweight='bold')

        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right') 

        # Show the plot
        plt.tight_layout()
        plt.show()

        logging.info("Bar Chart Plotted!!")

    def christmas_plot(self):

        # Define Holiday periods
        holiday_date = self.data[self.data['StateHoliday'] == 'c']['Date'].mode()[0]

        before_holiday_date = holiday_date - pd.Timedelta(days=7)
        after_holiday_date = holiday_date + pd.Timedelta(days=7)

        # Update the 'holidays' column based on conditions
        self.data['holidays'] = 'None'
        self.data.loc[(self.data['Date'] >= before_holiday_date) & (self.data['Date'] < holiday_date), 'holidays'] = 'Before'
        self.data.loc[self.data['Date'] == holiday_date, 'holidays'] = 'During'
        self.data.loc[(self.data['Date'] > holiday_date) & (self.data['Date'] <= after_holiday_date), 'holidays'] = 'After'

        logging.info("Plotting Bar chart... ")

        # Create the bar plot
        plt.figure(figsize=(12, 6))  # Set the figure size
        sns.barplot(data=self.data, x='holidays', y='Sales', palette='viridis')

        # Add title and labels
        plt.title("Sales Distribution During, Before, and After Christmass", fontsize=14, fontweight='bold')
        plt.xlabel('Holiday Period', fontsize=12, fontweight='bold')
        plt.ylabel('Sales', fontsize=12, fontweight='bold')

        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right') 

        # Show the plot
        plt.tight_layout()
        plt.show()

        logging.info("Bar Chart Plotted!!")

