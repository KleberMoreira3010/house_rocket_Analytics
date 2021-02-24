from typing import Container
import geopandas
import streamlit as st
import pandas as pd
import numpy as np
import folium as fl
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import plotly.express as px
from datetime import datetime

#Mudando layout do sistema
st.set_page_config(layout='wide' )


#Gerando cache dos dados
@st.cache(allow_output_mutation=True)

#Função busca dados
def get_data(path):
        data= pd.read_csv(path)
        return data


#Função busca geodados
@st.cache(allow_output_mutation=True)
def get_geofile(url):
        geofile=geopandas.read_file(url)
        return geofile

def set_feature(data):
#Criando uma varável
    data['price_m2']= data['price']/data['sqft_lot']
    return data


def overview_data(data):
    f_attributes=st.sidebar.multiselect('Enter columns', data.columns)
    f_zipcode = st.sidebar.multiselect( 'Enter zipcode', data['zipcode'].unique())


    #attributes + zipcode = Selecionar linhas e colunas
    #attributes = colunas
    #zipcode = linhas
    # 0 + 0 = dataset original

    #Modo de selecionar linhas e colunas
    if(f_zipcode!=[]) & (f_attributes!=[]):
        data=data.loc[data['zipcode'].isin(f_zipcode), f_attributes]

    elif  (f_zipcode!=[]) & (f_attributes==[]):
        data=data.loc[data['zipcode'].isin(f_zipcode), :]

    elif  (f_zipcode==[]) & (f_attributes!=[]):
        data=data.loc[:, f_attributes]

    else:
        data=data.copy()

    st.dataframe(data.head())

    #Modo que organiza as tabelas na página
    c1,c2 = st.beta_columns((1,1))

    st.title('Data Overview')


    #Métricas
    df1=data[['id','zipcode']].groupby('zipcode').count().reset_index()
    df2=data[['price','zipcode']].groupby('zipcode').mean().reset_index()
    df3=data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4=data[['price_m2','zipcode']].groupby('zipcode').mean().reset_index()

    #Criando o dataset (MERGE)
    m1=pd.merge(df1,df2, on='zipcode', how='inner')
    m2=pd.merge(m1,df3, on='zipcode', how='inner')
    df=pd.merge(m2,df4, on='zipcode', how='inner')

    df.columns=['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQFT_LIVING', 'PRICE/M2']

    c1.header('Metrics')
    c1.dataframe(df, height=600)


    #Statistc Descriptive

    num_attributes = data.select_dtypes(include=['int64', 'float64'])
    media=pd.DataFrame(num_attributes.apply(np.mean))
    mediana=pd.DataFrame(num_attributes.apply(np.median))
    std=pd.DataFrame(num_attributes.apply(np.std))
    max_=pd.DataFrame(num_attributes.apply(np.max))
    min_=pd.DataFrame(num_attributes.apply(np.min))

    df1=pd.concat([max_, min_,media, mediana, std], axis=1).reset_index()

    df1.columns=['ATTRIBUTES', 'MAX', 'MIN', 'MEAN', 'MEDIAN', 'STD']

    c2.header('Descriptive Analysis')
    c2.dataframe(df1, height=600)


    return None

def portfolio_density(data,geofile):
    # 1.5- Um mapa com a densidade de portfólio por região e também densidade de preço
    # ------------------------

    #  Densidade de Portfolio

    st.title('Overview por Região')

    c1, c2 = st.beta_columns((1, 1))
    c1.header('Densidade de Portfolio')

    df = data.sample(10)

    # Base Map - Folium
    densidade_map = fl.Map(location=[data['lat'].mean(),
                                    data['long'].mean()],
                        default_zoom_start=15)
    marker_cluster = MarkerCluster().add_to(densidade_map)
    for name, row in df.iterrows():
        fl.Marker( [row['lat'], row['long']],
                popup='Sold R${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format(row['price'],
                                                                                                                        row['date'],
                                                                                                                    row['sqft_living'],
                                                                                                                    row['bedrooms'],
                                                                                                                    row['bathrooms'],
                                                                                                                    row['yr_built'])).add_to(marker_cluster)

    with c1:
        folium_static(densidade_map)

    #  Densidade de Preço
    # ------------------------
    # Mapa de preço por região
    c2.header('Densidade de Preço')

    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['zip', 'price']
    df = df.sample(10)

    regiao_map = fl.Map(location=[data['lat'].mean(),
                                    data['long'].mean()],
                        default_zoom_start=15)

    #geofile = fl.load_geofile(url)
    geofile = geofile[geofile['ZIP'].isin( df['zip'].tolist() )]

    regiao_map.choropleth(data=df, geo_data=geofile, columns=['zip', 'price'], key_on='feature.properties.ZIP',
                        fill_color='YlOrRd',
                        fill_opacity=0.7,
                        line_opacity=0.2,
                        legend_name='MEDIA PREÇO')
    with c2:
        folium_static(regiao_map)

    ##########################################################
    #Distribuição dos imóveis por categoria
    ##########################################################

    data['date']=pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')



    return None


def commercial(data):
    st.sidebar.title('Commercial Options')

    st.title('Commercial Attributes')

    ####Every Price por Year

    ###Filter
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    st.sidebar.subheader('Select Max Year built')
    f_year_built=st.sidebar.slider('Year Built', min_year_built, max_year_built, min_year_built)



    #data select
    df = data.loc[data['yr_built'] < f_year_built]
    df=df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    fig=px.line(df, x='yr_built', y='price')


    #plot
    st.plotly_chart(fig, use_container_width=True)

    ####Every Price por day
    st.header('Average Price por day')
    st.sidebar.subheader('Select Max Date')

    #filter
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')

    f_date=st.sidebar.slider('Date', min_date, max_date, min_date)


    #data filtering
    data['date']=pd.to_datetime(data['date'])
    df=data.loc[data['date']<f_date]
    df=df[['date', 'price']].groupby('date').mean().reset_index()
    fig=px.line(df, x='date', y='price')


    #plot
    st.plotly_chart(fig, use_container_width=True)

    ###########
    #Histograma
    ###########
    st.header('Price Distribution')
    st.sidebar.subheader('Select Max Price')

    #filter
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    #datafiltering
    f_price=st.sidebar.slider('Price', min_price, max_price, avg_price)
    df=data.loc[data['price']<f_price]

    #plot
    fig=px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    return None

def attributes_distribution(data):
    ################################################
    #Distribuição dos imóveis por categorias fisicas
    ################################################

    st.sidebar.title('Attributes Options')
    st.title('House Attributes')


    #filter
    f_bedrooms=st.sidebar.selectbox('Max number of bedrooms',
                        sorted(set(data['bedrooms'].unique())))


    f_bathrooms=st.sidebar.selectbox('Max number of bathrooms',
                        sorted(set(data['bathrooms'].unique())))


    c1, c2 =st.beta_columns(2)


    # House per bedrooms
    c1.header('Houses per bathrooms')
    df=data['bedrooms'] < f_bedrooms
    fig=px.histogram(data, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # House per bathrooms
    c2.header('Houses per bathrooms')
    df=data['bathrooms'] < f_bathrooms
    fig=px.histogram(data, x='bathrooms', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)


    #filter
    f_floors=st.sidebar.selectbox('Max number of floors',
                        sorted(set(data['floors'].unique())))

    f_waterview = st.sidebar.checkbox('Only Houses Water View')

    c1,c2 = st.beta_columns(2)

    # House per floors
    c1.header('Houser per floor')
    df=data['floors']<f_floors

    #plot
    fig=px.histogram(data, x='floors', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # House per water view
    if f_waterview:
        df=data[data['waterfront']==1]
    else:
        df = data.copy()

    fig = px.histogram(df, x='waterfront', nbins=10)
    c2.plotly_chart(fig, use_Container_width=True)        

    return None
if __name__== '__main__':
    #ETL
    #data extration
    path='kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    
    data=get_data(path)
    geofile=get_geofile(url)


#transformation
data=set_feature(data)

overview_data(data)

portfolio_density(data, geofile)

commercial (data)

attributes_distribution(data)


