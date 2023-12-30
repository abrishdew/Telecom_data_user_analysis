import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
import seaborn as sns
import matplotlib as plt


class DataCleaner:
    
    def drop_duplicate(self, telecom_df: pd.DataFrame) -> pd.DataFrame:
        
        #Drop Duplicated rows
        
        telecom_df.drop_duplicates(inplace=True)

        return telecom_df
    def convert_to_datetime(self, telecom_df: pd.DataFrame) -> pd.DataFrame:
        
        #Convert datatype to datetime
        

        telecom_df[['Start','End']] = telecom_df[['Start','End']].apply(pd.to_datetime)

        return telecom_df

    def convert_to_string(self, telecom_df: pd.DataFrame) -> pd.DataFrame:
        
        #Convert datatype to string
        
        telecom_df[['Bearer_Id', 'IMSI', 'MSISDN/Number', 'IMEI','Handset_Type']] = telecom_df[['Bearer_Id', 'IMSI', 'MSISDN/Number', 'IMEI','Handset_Type']].astype(str)

        return telecom_df

    def remove_whitespace_column(self, telecom_df: pd.DataFrame) -> pd.DataFrame:
        
        #Remove White Spaces

        telecom_df.columns = [col.replace(' ', '_') for col in telecom_df.columns]
        
        return telecom_df

    
    def percent_missing(self, telecom_df: pd.DataFrame) -> pd.DataFrame:

        # Calculate total number of cells in dataframe
        totalCells = np.product(telecom_df.shape)

        # Count number of missing values per column
        missingCount = telecom_df.isnull().sum()

        # Calculate total number of missing values
        totalMissing = missingCount.sum()

        # Calculate percentage of missing values
        percentage_miss = round(((totalMissing/totalCells) * 100), 2)

        return percentage_miss

    def fill_missing_values_numeric(self, telecom_df: pd.DataFrame, method: str,columns: list =None) -> pd.DataFrame:
        
        # Fill Missing numeric values with Median or Mean

        if(columns==None):
            numeric_columns = self.get_numerical_columns(telecom_df)
        else:
            numeric_columns=columns

        if method == "median":
            for col in numeric_columns:
                telecom_df[col].fillna(telecom_df[col].median(), inplace=True)

        elif method == "mean":
            for col in numeric_columns:
                telecom_df[col].fillna(telecom_df[col].mean(), inplace=True)
        else:
            print("Method unknown")
        
        return telecom_df
    
    def fill_missing_values_categorical(self, telecom_df: pd.DataFrame, method: str) -> pd.DataFrame:
      
      # Fill Missing Categorical Values with ffill, bfill or mode
        categorical_columns = telecom_df.select_dtypes(include=['object','datetime64[ns]']).columns

        if method == "ffill":

            for col in categorical_columns:
                telecom_df[col] = telecom_df[col].fillna(method='ffill')

            return telecom_df

        elif method == "bfill":

            for col in categorical_columns:
                telecom_df[col] = telecom_df[col].fillna(method='bfill')

            return telecom_df

        elif method == "mode":
            
            for col in categorical_columns:
                telecom_df[col] = telecom_df[col].fillna(telecom_df[col].mode()[0])

            return telecom_df
        else:
            print("Method unknown")
            return telecom_df


    def save_data(self, telecom_df: pd.DataFrame, data_path:str,index:bool = False) -> None:
       
       "Save the new filled and cleaned data"

       try:
           telecom_df.to_csv(data_path,index=index)
           print("Complete!")
           
       except Exception as e:
           print(f"Saving failed {e}")   



# Function to calculate missing values by column
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # dtype of missing values
    mis_val_dtype = df.dtypes

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values', 2: 'Dtype'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

def format_float(value):
    return f'{value:,.2f}'

def find_agg(df:pd.DataFrame, agg_column:str, agg_metric:str, col_name:str, top:int, order=False )->pd.DataFrame:

    new_df = df.groupby(agg_column)[agg_column].agg(agg_metric).reset_index(name=col_name).\
                        sort_values(by=col_name, ascending=order)[:top]

    return new_df

def convert_bytes_to_megabytes(df, bytes_data):
    """
        This function takes the dataframe and the column which has the bytes values
        returns the megabytesof that value

        Args:
        -----
        df: dataframe
        bytes_data: column with bytes values

        Returns:
        --------
        A series
    """

    megabyte = 1*10e+5
    df[bytes_data] = df[bytes_data] / megabyte
    return telecom_df[bytes_data]

def fix_outlier(telecom_df, column):
    telecom_df[column] = np.where(telecom_df[column] > telecom_df[column].quantile(0.95), telecom_df[column].median(),telecom_df[column])

    return telecom_df[column]




###################################PLOTTING FUNCTIONS###################################

def plot_hist(telecom_df:pd.DataFrame, column:str, color:str)->None:
    # plt.figure(figsize=(15, 10))
    # fig, ax = plt.subplots(1, figsize=(12, 7))
    sns.displot(data=telecom_df, x=column, color=color, kde=True, height=7, aspect=2)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    plt.show()

def plot_count(telecom_df:pd.DataFrame, column:str) -> None:
    plt.figure(figsize=(12, 7))
    sns.countplot(data=telecom_df, x=column)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    plt.show()

def plot_bar(telecom_df:pd.DataFrame, x_col:str, y_col:str, title:str, xlabel:str, ylabel:str)->None:
    plt.figure(figsize=(12, 7))
    sns.barplot(data = telecom_df, x=x_col, y=y_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.yticks( fontsize=14)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.show()

def plot_heatmap(telecom_df:pd.DataFrame, title:str, cbar=False)->None:
    plt.figure(figsize=(12, 7))
    sns.heatmap(telecom_df, annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.2f', linewidths=.7, cbar=cbar )
    plt.title(title, size=18, fontweight='bold')
    plt.show()

def plot_box(telecom_df:pd.DataFrame, x_col:str, title:str) -> None:
    plt.figure(figsize=(12, 7))
    sns.boxplot(data = telecom_df, x=x_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.show()

def plot_box_multi(telecom_df:pd.DataFrame, x_col:str, y_col:str, title:str) -> None:
    plt.figure(figsize=(12, 7))
    sns.boxplot(data = telecom_df, x=x_col, y=y_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.yticks( fontsize=14)
    plt.show()

def plot_scatter(telecom_df: pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str) -> None:
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data = df, x=x_col, y=y_col, hue=hue, style=style)
    plt.title(title, size=20)
    plt.xticks(fontsize=14)
    plt.yticks( fontsize=14)
    plt.show()


pd.options.display.float_format = format_float