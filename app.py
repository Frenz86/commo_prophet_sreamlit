import investpy
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
from matplotlib import pyplot as plt
from fbprophet import Prophet
from fbprophet.plot import plot_plotly

today=datetime.today().strftime('%d/%m/%Y')
commodity_ = investpy.commodities.get_commodities_list(group=None)

asset = st.sidebar.selectbox('Click below to select a new asset',
                                commodity_)

commo = investpy.get_commodity_historical_data(commodity=asset, 
                                            from_date='01/01/1980', 
                                            to_date=today,
                                            as_json=False,
                                            order='ascending')

data = commo.copy()
data.index.name = None

section = st.sidebar.slider('Number of observation', 
                            min_value=30,
                            max_value=min([2000, data.shape[0]]),
                            value=500,  
                            step=10)

st.dataframe(data)


data2 = data[-section:]['Close'].to_frame('Price')
data2['Date'] = data2.index

fig2 = px.line(data2,x='Date', y="Price",title = 'the Title ')
fig2.update_xaxes(
    rangeslider_visible= True,
    rangeselector=dict(
                        buttons = list([
                        dict(count = 3,label = '1y',step='year',stepmode = "backward"),
                        dict(count = 9,label = '3y',step='year',stepmode = "backward"),
                        dict(count = 15,label = '5y',step='year',stepmode = "backward"),
                        dict(step= 'all')
                            ])        
                        )
                )
st.plotly_chart(fig2)


data2 = (
    data2['Price']
    .dropna()
    .to_frame()
    .reset_index()
    .rename(columns={"index": "ds", 'Price': "y"})
)

windows = st.sidebar.slider('prediction window', 
                            min_value=80,
                            max_value=250,
                            value=90,  
                            step=1)


model = Prophet()
model.fit(data2)
future = model.make_future_dataframe(periods=windows)
forecast = model.predict(future)


fig = plot_plotly(model, forecast)
fig.update_layout(
    title='title', yaxis_title=asset , xaxis_title="Date",
)

st.write("# Forecast Prices")
st.plotly_chart(fig)

##########################################
model.plot_components(forecast)
#plt.title("Global Products")
plt.legend()
st.pyplot()