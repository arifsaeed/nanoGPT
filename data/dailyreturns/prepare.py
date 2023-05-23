import os
import csv
import pandas as pd
import json
import torch
import pickle
import datetime as dt
from torch.utils.data import Dataset


class DailyLSEDataset(Dataset):
    def __init__(self):
        self.start_date = dt.datetime(2012, 2, 1)
        self.end_date = dt.datetime(2023, 2, 1)
        self.date_length=7
        self.columns_to_convert = ['Low_norm', 'Open_norm', 'Volume_norm', 'High_norm', 'Adjusted Close_norm']
        self.label_columns=['Low_norm','High_norm','Adjusted Close_norm']
        self.path = '/home/arif/Documents/LLM/sandpit/historic_data/csv'
        self.company_mapping_file ='/home/arif/Documents/LLM/sandpit/historic_data/company_mapping.pkl'
        self.company_mapping=self.load_pickle_file(self.company_mapping_file)
        self.date_range=self.create_date_range()
        self.dataset=self.create_dateset()  

    def normalize(self,df,cols):
        for col in cols:
            max_value = df[col].max()
            min_value = df[col].min()
            df[col +'_norm'] = (df[col] - min_value) / (max_value - min_value)

    def create_date_range(self):
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        df = pd.DataFrame(date_range, columns=['Date'])
        return df


    def create_company_ids(self,df):
        companies = df["Company_name"].unique().tolist()
        company_lookup={}
        for idx,company in enumerate(companies):
            company_lookup[company]=idx
        return company_lookup

    def get_train_label_dates(self,date):
        index = self.date_range.index[self.date_range['Date']==date][0]
        dates=[]
        if index<self.date_length:
            index==self.date_length
        if index==len(self.date_range)-1:
            index = index -1

        for i in range(self.date_length):
            dates.append(self.date_range.iloc[index-i]['Date'])

        return {'x_dates':dates,'y_date':self.date_range.iloc[index+1]['Date']}

    def create_dateset(self):
        master_created =False
        for filename in os.listdir(self.path):
            df_temp=pd.read_csv(self.path + '/' + filename)
            df_temp['Date'] = pd.to_datetime(df_temp['Date'], format='%d-%m-%Y')
            names = filename.split('.')
            df_merged = pd.merge(self.date_range, df_temp, on='Date', how='left')
            df_merged['Company_name'] = names[0]
            df_merged.fillna(0, inplace=True)
            if not master_created:
                df=df_merged
                master_created=True
            else:
                df=pd.concat([df,df_merged])
        self.normalize(df,['Low','Open','Volume','High','Adjusted Close'])
        df['Company']= df["Company_name"].map(self.company_mapping)
        df=df.reset_index()
        return df

    def load_pickle_file(self,file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    
    def to_tensor(self,dates):
        #dates = [dt.datetime(2023,1,16),dt.datetime(2023,1,17),dt.datetime(2023,1,18),dt.datetime(2023,1,19),dt.datetime(2023,1,20)]
        
        tensor_initialised =False
        for date in dates:
            filtered_df = self.dataset[(self.dataset['Date'] == date)][['Company','Low_norm', 'Open_norm','Volume_norm','High_norm','Adjusted Close_norm']]
            if not tensor_initialised:
                tensor =torch.tensor(filtered_df[self.columns_to_convert].values)
                tensor_initialised=True
            else:
                tensor = torch.cat((tensor,torch.tensor(filtered_df[self.columns_to_convert].values)),dim=1)
        return tensor
    
    def label_to_tensor(self,today,tomorrow,company):
        tomorrow_filter = self.dataset['Date'] == tomorrow
        today_filter = self.dataset['Date'] == today
        company_filter = self.dataset['Company'] == company
        tomorrow_df = self.dataset[tomorrow_filter & company_filter]#[['Company','Low_norm', 'Open_norm','Volume_norm','High_norm','Adjusted Close_norm']]
        today_df=self.dataset[today_filter & company_filter]
        daily_return = (tomorrow_df['Adjusted Close'].values[0] - today_df['Adjusted Close'].values[0])/today_df['Adjusted Close'].values[0]

        return torch.tensor(daily_return)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        idx_date=self.dataset.iloc[idx]['Date']
        company = self.dataset.iloc[idx]['Company']
        all_dates=self.get_train_label_dates(idx_date)
        x_tensor = self.to_tensor(all_dates['x_dates'])
        daily_return=self.label_to_tensor(all_dates['x_dates'][0],all_dates['y_date'],company)

        return (x_tensor,daily_return)

if __name__ == '__main__':
    lse=DailyLSEDataset()
    lse.__getitem__(356)
    