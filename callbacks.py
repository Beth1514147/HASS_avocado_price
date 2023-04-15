'''' 
Functions of callbacks to call results
'''

import pandas as pd 
from prophet import Prophet 
from prophet.plot import add_changepoints_to_plot
from datetime import datetime, timedelta
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# # Return results in price prediction:
# def show_results3(df,country,year_predict,topn_country):

#     train_data, test_data, model_organic_newyork, forcast_organic_newyork = prophet_forecast_yearly_seasonality(df_organic_newyork, train_start, train_end, test_end, 'revenue', 'date', 5)
    


def show_price_prediction(price_model,month,Volume_4046,Volume_4225,Volume_4770):
    total_vol = Volume_4046 + Volume_4225 + Volume_4770
    x_pred = np.array([month,total_vol], dtype = np.float64).reshape(1, -1)
    y_pred = price_model.predict(x_pred)
    # st.write('Gia bo trung binh du doan: ', y_pred)
    return y_pred


def show_price_prediction2(dataframe, price_model):
    dataframe['total_vol'] = dataframe['4046'] + dataframe['4225'] + dataframe['4770']
    x_pred = dataframe[['month','total_vol']]
    y_pred = price_model.predict(x_pred)
    df = pd.concat([dataframe,pd.DataFrame(y_pred,columns = ['y_predict'])], axis =1 )
    # st.write('Gia bo trung binh du doan: ', y_pred)
    return df


def prophet_forecast_yearly_seasonality(df, price_col, date_col, num_years_forcast=5):
    df.index = pd.to_datetime(df[date_col])
    df = df[[date_col, price_col]]
    df.columns = ['ds','y']
    train_start = datetime(2015, 1, 4)
    train_end = datetime(2017, 3, 15)
    test_end = datetime(2018, 3, 25)
    train_data = df['2015-01-04':'2017-03-15'].reset_index().drop_duplicates()
    test_data = df['2017-03-22':].reset_index().drop_duplicates()
    # Build model
    model_prophet = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
    # fit model 
    model_prophet.fit(train_data)
    # predict new values
    num_years = num_years_forcast 
    weeks = pd.date_range('2017-03-15','2023-03-25', freq='W').strftime("%Y-%m-%d").tolist()    
    future = pd.DataFrame(weeks)
    future.columns = ['ds']
    future['ds'] = pd.to_datetime(future['ds'])
    # Use the model to make forecast
    forecast = model_prophet.predict(future)
    return model_prophet, forecast



def show_results3(country,years,df,type):
    df_single = df[(df.region == country) & (df.type == type)]
    model_organic, forcast_organic = prophet_forecast_yearly_seasonality(df_single,'AveragePrice', 'Date', years)
    # fig = model_organic.plot(forcast_organic)
    # fig = add_changepoints_to_plot(fig.gca(), model_organic, forcast_organic)
    fig2 = plt.figure(figsize=(15,8))
    plt.plot(df_single.index, df_single['AveragePrice'], label='Actual')
    plt.plot(forcast_organic['ds'], forcast_organic['yhat'], label='Prediction', color='red')
    plt.title('AveragePrice of {} Avocado Prediction in New York for next {} years'.format(type,years), fontdict={'weight':'bold','fontsize':15, 'color':'g'})

    return fig2
