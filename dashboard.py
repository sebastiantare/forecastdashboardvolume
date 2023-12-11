import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import os

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

df = pd.DataFrame()
max_date = None
gr_df = pd.DataFrame()
gr_ad_df = pd.DataFrame()
previous_for_df = pd.DataFrame()
selected_tasks = []

tab1, tab2 = st.tabs(["Task Volume Forecast", "Growth Summary"])

@st.cache_data
def load_and_forecast(file):
    if file.type == 'text/csv':
        df = pd.read_csv(file, usecols=[0, 1, 2])
    elif file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        df = pd.read_excel(file, sheet_name=0, usecols=[0, 1, 2])

    df.columns = ['Tasks', 'Date', 'Volume']
    df = df[df['Tasks'].notnull()]
    df['Date'] = pd.to_datetime(df['Date'])
    df['Volume'].fillna(0, inplace=True)

    df.sort_values(by=['Date', 'Tasks'], inplace=True, ascending=True)

    tasks = df['Tasks'].unique()
    max_date = df['Date'].max()
    min_date = df['Date'].min()
    months_difference = ((max_date.year - min_date.year) * 12 + max_date.month - min_date.month) + 1
    date_range = pd.date_range(start=min_date, periods=months_difference, freq='MS') #+ pd.DateOffset(months=1)
    date_range_df = pd.DataFrame({'Date': date_range})

    merged_df = pd.DataFrame()

    for task in tasks:
        imputed_df = pd.merge(date_range_df, df[df['Tasks'] == task], how='left', on='Date')
        imputed_df['Tasks'].fillna(task, inplace=True)
        imputed_df['Volume'].fillna(0.0, inplace=True)
        merged_df = pd.concat([merged_df, imputed_df], axis=0)
    print(merged_df)

    df = merged_df.copy()

    for task in tasks:
        task_data = df[df['Tasks'] == task]

        steps = 6
        ### Change this to select a different model to forecast data.
        forecast = arima_forecast(task_data)
        #forecast = sarima_forecast(task_data)
        #forecast = ets_forecast(task_data)

        future_dates = pd.date_range(
            start=max_date + pd.DateOffset(months=1), periods=steps, freq='MS')
        forecast_df = pd.DataFrame(
            {'Date': future_dates, 'Tasks': task, 'Volume': forecast.values})

        df = pd.concat([df, forecast_df], ignore_index=True)

    df = df.sort_values(by='Date',  ascending=True).reset_index(drop=True)
    df['Volume'] = df['Volume'].astype(int)

    return df, max_date

@st.cache_data
def load_forecast_file(file):
    if file.type == 'text/csv':
        df = pd.read_csv(file, usecols=[0, 1, 2])
    elif file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        df = pd.read_excel(file, sheet_name=0, usecols=[0, 1, 2])

    df.columns = ['Tasks', 'Date', 'Volume']
    df = df[df['Tasks'].notnull()]
    df['Date'] = pd.to_datetime(df['Date'])
    df['Volume'].fillna(0, inplace=True)

    df = df.sort_values(by=['Tasks', 'Date'], ascending=True).reset_index(drop=True)
    df['Volume'] = df['Volume'].astype(int)
    print(df)

    return df

def arima_forecast(df, order=(5, 1, 2)):
    forecast = None
    try:
        if len(df['Volume']) < 10:
            # Use a simpler ARIMA model for smaller datasets
            model = ARIMA(df['Volume'], order=(2, 1, 0))
        else:
            # Use the original ARIMA model for larger datasets
            model = ARIMA(df['Volume'], order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=6)
    except:
        model = ARIMA(df['Volume'], order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=6)

    return forecast

def sarima_forecast(df, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(df['Volume'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    forecast = model_fit.get_forecast(steps=6)
    forecast_values = forecast.predicted_mean
    return forecast_values

def ets_forecast(df, trend='add', seasonal='add', seasonal_periods=12):
    model = ExponentialSmoothing(df['Volume'], trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=6)
    return forecast

def export_to_csv(df1, df2, df3, filename='output_file.xlsx'):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name='Forecast', index=0)
        df2.to_excel(writer, sheet_name='Seasonality', index=0)
        df3.to_excel(writer, sheet_name='Growth', index=0)
    st.success(f'Data exported to {filename}.')

    # Get the file name and mimetype
    file_name = os.path.basename('./' + filename)

    # Create the download button
    st.download_button(
        label="Click to Download File",
        data=open('./'+filename, "rb").read(),
        file_name=file_name,
    )

# Streamlit UI
with tab1:
    st.title('Forecasting Dashboard')

    uploaded_file = st.file_uploader("Upload your file", type=['csv', 'xlsx'])
    forecast_file = st.file_uploader("Upload previous forecast to compare", type=['csv', 'xlsx'])

    if forecast_file:
        for_df = load_forecast_file(forecast_file)
        previous_for_df = for_df.copy()

    if uploaded_file:
        df, max_date = load_and_forecast(uploaded_file)

        unique_tasks = df['Tasks'].unique()
        unique_tasks.sort()
        unique_tasks = list(unique_tasks)

        all_selected = st.checkbox('Select All Tasks')
        
        if all_selected:
            selected_tasks = unique_tasks

        selected_tasks = st.multiselect('**Select Tasks**', unique_tasks, default=selected_tasks)
        print(selected_tasks)
        
        if len(selected_tasks) > 0:
            #filtered_df = df[df['Tasks'] == selected_task]
            filtered_df = df[df['Tasks'].isin(selected_tasks)].groupby(['Date'])['Volume'].sum().reset_index()
            original_data = filtered_df[filtered_df['Date'] <= max_date]
            forecasted_data = filtered_df[filtered_df['Date'] >= max_date]

            if len(previous_for_df) > 0:
                #previous_forecast = previous_for_df[previous_for_df['Tasks'] == selected_task]
                previous_forecast = previous_for_df[previous_for_df['Tasks'].isin(selected_tasks)].groupby(['Date'])['Volume'].sum().reset_index()

            months_names = forecasted_data['Date'].dt.month_name()
            seasonal_data = pd.DataFrame(columns=months_names)
            seasonal_data.loc[0] = [0.0] * len(months_names)

            """
            **Adjust Seasonality (for all Tasks)**
            """

            edited_seasonal_data = st.data_editor(seasonal_data[seasonal_data.columns[1:]], width=800)
            adjusted_forecasted_data = forecasted_data.copy().reset_index(drop=True)

            if not adjusted_forecasted_data.empty:
                adjusted_forecasted_data.iloc[0, adjusted_forecasted_data.columns.get_loc('Volume')] = original_data['Volume'].iloc[-1]

                for i in adjusted_forecasted_data.index[1:]:
                    adjusted_forecasted_data.iloc[i, adjusted_forecasted_data.columns.get_loc('Volume')] *= 1 + (
                        edited_seasonal_data.iloc[:, i-1] / 100)

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=original_data['Date'], y=original_data['Volume'], mode='lines+markers', name='Original', line=dict(color='#3498db')))

            fig.add_trace(go.Scatter(x=forecasted_data['Date'], y=forecasted_data['Volume'], mode='lines+markers', name='Forecast', line=dict(color='#2c3e50')))

            if len(previous_for_df) > 0:
                fig.add_trace(go.Scatter(x=previous_forecast['Date'], y=previous_forecast['Volume'], mode='lines+markers', name='Previous Forecast', line=dict(color='#f39c12')))

            fig.add_trace(go.Scatter(x=adjusted_forecasted_data['Date'], y=adjusted_forecasted_data['Volume'], mode='lines+markers', name='Adjusted Forecast', line=dict(color='#e74c3c')))

            new_df = df[df['Date'] >= max_date].copy()

            new_df['Growth'] = new_df.groupby(['Tasks'])['Volume'].pct_change().fillna(0)

            # Add the 1 for the last observation
            mul_values = [1.0] + list(1 + (edited_seasonal_data.values[0]/100))
            result_df = new_df.groupby('Tasks').apply(lambda group: group['Volume'] * mul_values).reset_index(level=0)
            
            # Add dates based on the same indexes of the data
            final_df = pd.merge(result_df.reset_index(), df[['Date']].reset_index(), on='index', how='left')
            final_df.drop(columns=['index'], inplace=True)
            final_df['Date'] = final_df['Date'].dt.strftime('%Y-%m-%d')
            new_order = ['Tasks', 'Date', 'Volume']
            final_df = final_df[new_order]

            # result_df['Adjusted Growth'] = result_df.groupby(['Tasks'])['Volume'].pct_change().fillna(0)
            # gr_ad_df = result_df.groupby('Tasks')[['Adjusted Growth']].mean()
            first_last_volume = result_df.groupby('Tasks')['Volume'].agg(['first', 'last'])
            gr_ad_df['Adjusted Growth'] = (
                (first_last_volume['last'] - first_last_volume['first'])/first_last_volume['first'])

            gr_ad_df = gr_ad_df.sort_values(by='Adjusted Growth', ascending=False)
            gr_ad_df['Adjusted Growth'] = (
                gr_ad_df['Adjusted Growth'] * 100).round(2).astype(str) + '%'

            # gr_df = new_df.groupby('Tasks')[['Growth']].mean()
            first_last_volume = new_df.groupby(
                'Tasks')['Volume'].agg(['first', 'last'])
            gr_df['Growth'] = ((first_last_volume['last'] -
                            first_last_volume['first'])/first_last_volume['first'])

            gr_df = gr_df.sort_values(by='Growth', ascending=False)
            gr_df['Growth'] = (gr_df['Growth'] * 100).round(2).astype(str) + '%'

            gr_df = pd.merge(gr_df, gr_ad_df, on='Tasks', how='inner')

            df_concat = pd.concat(
                [original_data, adjusted_forecasted_data.iloc[1:]], ignore_index=True)

            mean_volume = df_concat['Volume'].mean()
            std_dev_volume = df_concat['Volume'].std()

            upper_limit = mean_volume + 2 * std_dev_volume
            lower_limit = mean_volume - 2 * std_dev_volume

            control_limit_method = st.selectbox(
                'Control Limit Method', ['Fixed', 'Std Deviations'])

            if control_limit_method == 'Fixed':
                upper_limit = mean_volume + 2 * std_dev_volume
                lower_limit = mean_volume - 2 * std_dev_volume
            elif control_limit_method == 'Std Deviations':
                std_dev_multiplier = st.slider(
                    'Std Deviation Multiplier', min_value=1, max_value=5, value=2)
                upper_limit = mean_volume + std_dev_multiplier * std_dev_volume
                lower_limit = mean_volume - std_dev_multiplier * std_dev_volume

            fig.add_shape(
                type='line',
                x0=filtered_df['Date'].iloc[0],
                x1=filtered_df['Date'].iloc[-1],
                y0=mean_volume,
                y1=mean_volume,
                line=dict(color='black', dash='dash'),
                name='Mean'
            )

            fig.add_shape(
                type='line',
                x0=filtered_df['Date'].iloc[0],
                x1=filtered_df['Date'].iloc[-1],
                y0=upper_limit,
                y1=upper_limit,
                line=dict(color='blue', dash='dash'),
                name='Upper Control Limit'
            )

            fig.add_shape(
                type='line',
                x0=filtered_df['Date'].iloc[0],
                x1=filtered_df['Date'].iloc[-1],
                y0=lower_limit,
                y1=lower_limit,
                line=dict(color='blue', dash='dash'),
                name='Lower Control Limit'
            )

            fig.update_layout(
                annotations=[
                    dict(
                        x=df['Date'].max(),
                        y=upper_limit,
                        xref="x",
                        yref="y",
                        text=f"UCL: {round(upper_limit, 3)}",
                        showarrow=True,
                        arrowhead=0,
                        ax=50,
                        ay=0
                    ),
                    dict(
                        x=df['Date'].max(),
                        y=mean_volume,
                        xref="x",
                        yref="y",
                        text=f"MEAN: {round(mean_volume,3)}",
                        showarrow=True,
                        arrowhead=0,
                        ax=50,
                        ay=0
                    ),
                    dict(
                        x=df['Date'].max(),
                        y=lower_limit,
                        xref="x",
                        yref="y",
                        text=f"LCL: {round(lower_limit, 3)}",
                        showarrow=True,
                        arrowhead=0,
                        ax=50,
                        ay=0
                    )
                ]
            )

            st.plotly_chart(fig)

            # Adjusting dataframe display width
            """
            **Forecast Data (adjusted)**
            """
            df_concat['Date'] = pd.to_datetime(df_concat['Date']).dt.strftime('%Y-%m-%d')
            st.dataframe(df_concat.reset_index(drop=True), width=800)  # Adjust the width as needed

            # Add a button to trigger the export
            if st.button('Export All Tasks Forecast to CSV'):
                export_to_csv(final_df, edited_seasonal_data, gr_df.reset_index())

with tab2:
    """
    **Growth**
    """

    st.dataframe(gr_df, width=800, height=800)  # Adjust the width as needed