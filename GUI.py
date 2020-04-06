from PyQt5.QtCore import QDate, QTime, Qt, pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QToolTip, QMainWindow, QLineEdit, QLabel, QVBoxLayout
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView as QWebView
import pyqtgraph as pg
from numpy import array
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
import folium
import googlemaps
import numpy as np
import requests
import folium
from folium.plugins import HeatMap
from PyQt5 import QtWidgets, uic
flag= 0
raw_seq_cases = [580, 845, 1317, 2015, 2800, 4581, 6058, 7813,
     9823, 11950, 14553, 17391,20630, 24545, 28266, 31439, 34876,
     37552, 40553,43099, 45134, 59287, 64438, 67100, 69197, 71329,
     73332, 75184, 75700,76677, 77673, 78651, 79205, 80087, 80828, 
     81820, 83112,84615, 86604, 88585, 90443, 93016, 95314, 98425,
     102050, 106099, 109991,114381, 118948, 126948, 134576, 145483,
     156653, 169593,182490, 198238, 218822, 244933, 275597, 305036,
    337000, 378000, 422000, 471000, 531000, 596000, 663000, 723000,
    784000, 858000, 935000, 1010000]
raw_seq_cases_init = []
x_axis_cases = []
for i in range(len(raw_seq_cases)):
    raw_seq_cases[i] = raw_seq_cases[i]/ 2000000
    raw_seq_cases_init.append(raw_seq_cases[i])
for i in range(len(raw_seq_cases)):
    x_axis_cases.append(i)

raw_seq_recovery = [58.20,60.15,64.83,67.49,71.00,74.21,76.75,78.50,79.89,81.16,82.60,83.12,
           84.30,85.11,86.08,87.16,87.87,88.50,89.85,89.85,90.20,90.52,91.05,91.58,
           92.14,92.74,93.10,93.43,93.67,93.92,94.09,94.22,94.26,94.36,94.21,94.09,
           93.94,93.65,93.39,93.04,92.87,92.24,91.75,91.19,90.51,89.78,88.94,88.01,
           87.07, 86.07, 85.21, 84.29, 83.79, 82.95, 82.14, 81.62, 81.41, 80.81, 80.44,
           80.44, 79.95]

raw_seq_recovery_init = []
x_axis_recovery = []
for i in range(len(raw_seq_recovery)):
    raw_seq_recovery[i] = raw_seq_recovery[i]/ 100
    raw_seq_recovery_init.append(raw_seq_recovery[i])
for i in range(len(raw_seq_recovery)):
    x_axis_recovery.append(i)

n_features = 1

ip_request = requests.get('https://get.geojs.io/v1/ip.json')
my_ip = ip_request.json()['ip']
geo_request = requests.get('https://get.geojs.io/v1/ip/geo/' +my_ip + '.json')
geo_data = geo_request.json()
print({'latitude': geo_data['latitude'], 'longitude': geo_data['longitude']})
lat = geo_data['latitude']
long = geo_data['longitude']
m = folium.Map(location=[lat, long], tiles='cartodbpositron',
               min_zoom=1, max_zoom=20, zoom_start=13)
hospital=pd.read_csv('address.csv')
print("Done")
hospitalicon=folium.features.CustomIcon('hplus.png',icon_size=(50,50))
folium.Circle(location=[lat,long],radius=5000,popup="Area near me",fill_color='#3186cc').add_to(m)
for i in range(0,len(hospital)):
    folium.Marker([hospital.iloc[i]['lat'],hospital.iloc[i]['long']],popup=hospital.iloc[i]['Hospital'],tooltip="<strong>{}</strong>".format(hospital.iloc[i]['Hospital']),icon=folium.Icon( icon='plus',color='red')).add_to(m)

m.save('map0.html')
dirpath = os.getcwd()
filepath = dirpath+'/map0.html'
print("File Path: ", filepath)
df=pd.read_csv('report.csv')
map1 = folium.Map(location=(df['lat'].mean(),df['lon'].mean()),tiles='cartodbdark_matter', zoom_start=7)
for i in range(0,len(df)):
    if(df['infected'][i]) > 0:
       
        for j in range(0,df['infected'][i]):
            folium.Circle(location=(df['lat'][i]+np.random.uniform(-0.1,0.01),df['lon'][i]+np.random.uniform(-0.01,0.1)),radius=3000,tooltip=df['infected'][i],
                          popup="<strong>{}</strong>".format(df['District'][i]),
                          line_color='red',color='red',fill_color='red').add_to(map1)           

map1.save('map1.html')
df=pd.read_csv('report.csv')
map3 = folium.Map(location=(df['lat'].mean(),df['lon'].mean()),tiles='cartodbdark_matter', zoom_start=7)

for i in range(0,len(df)):
    if(df['infected'][i]) > 0:
       
        for j in range(0,df['infected'][i]):
           
            folium.Circle(location=(df['lat'][i]+np.random.uniform(-0.1,0.01),df['lon'][i]+np.random.uniform(-0.01,0.1)),radius=30,tooltip=df['infected'][i],
                          popup="<strong>{}</strong>".format(df['District'][i]),
                          line_color='red',color='white',fill_color='red').add_to(map3)


df['lon']=df['lon'].astype(float)
df['lat']=df['lat'].astype(float)
heat_df=df[df['infected']>0]


heat_data=[]
for index,row in heat_df.iterrows():
    for i in range(0,row['infected']):
        heat_data.append([row['lat']+np.random.uniform(-0.1,0.01),row['lon']+np.random.uniform(-0.1,0.01)])
       
# Plot it on the map
HeatMap(heat_data).add_to(map3)
map3.save('map3.html')



def on_predict_recovery():
    global x_axis_recovery
    global raw_seq_recovery
    n_steps = 12
    model1 = load_model(dirpath+'/recovery.h5')
    new_x_axis = len(raw_seq_recovery)-1
    x_axis_values = [new_x_axis]
    y_axis_values = [raw_seq_recovery[new_x_axis]]
    for i in range(10): 
        x_input = raw_seq_recovery[(len(raw_seq_recovery)-n_steps):]
        x_input = np.array(x_input)
        x_input = x_input.reshape((1,n_steps, n_features))
        new_y = np.array(model1.predict(x_input, verbose = 0))
        print(new_y[0][0])
        raw_seq_recovery.append(new_y[0][0])
        #print("Len of raw seq:", len(raw_seq))
        new_x_axis += 1
        x_axis_values.append(new_x_axis)
        y_axis_values.append(new_y[0][0])

    pen = pg.mkPen(color = 'r')
    y_axis_values_plot = []
    for i in range(len(y_axis_values)):
        y_axis_values_plot.append(y_axis_values[i]*100)
    graph2.plot(x_axis_values, y_axis_values_plot, pen = pen, symbol = 'd')
    
def on_predict_cases():
    global x_axis_cases
    global raw_seq_cases
    print("Predict Cases")
    model = load_model(dirpath+'/cases.h5')
    new_x_axis = len(raw_seq_cases)-1
    n_steps = 12
    x_axis_values = [new_x_axis]
    y_axis_values = [raw_seq_cases[new_x_axis]]
    for i in range(10): 
        x_input = raw_seq_cases[(len(raw_seq_cases)-n_steps):]
        x_input = np.array(x_input)
        x_input = x_input.reshape((1,n_steps, n_features))
        new_y = np.array(model.predict(x_input, verbose = 0))
        print(new_y[0][0])
        raw_seq_cases.append(new_y[0][0])
        #print("Len of raw seq:", len(raw_seq))
        new_x_axis += 1
        x_axis_values.append(new_x_axis)
        y_axis_values.append(new_y[0][0])

    pen = pg.mkPen(color = 'r')
    y_axis_values_plot = []
    for i in range(len(y_axis_values)):
        y_axis_values_plot.append(y_axis_values[i]*2000)
    graph1.plot(x_axis_values, y_axis_values_plot, pen = pen, symbol = 'd')
    on_predict_recovery()

def on_reset():
    global x_axis_cases
    global raw_seq_cases_init
    global y_axis_recovery
    global raw_seq_recovery_init
    global raw_seq_cases
    global raw_seq_recovery

    raw_seq_recovery = []
    raw_seq_cases = []
    
    a = x_axis_cases
    b= raw_seq_cases_init
    raw_seq_cases_plot = []
    for i in range(len(raw_seq_cases_init)):
        raw_seq_cases.append(raw_seq_cases_init[i])
        raw_seq_cases_plot.append(2000*raw_seq_cases_init[i])    
    pen = pg.mkPen(color = 'g', width = 3)
    graph1.clear()
    graph1.plot(a,raw_seq_cases_plot, pen = pen, symbol = '+')

    a = x_axis_recovery
    b= raw_seq_recovery_init
    raw_seq_recovery_plot = []
    for i in range(len(raw_seq_recovery_init)):
        raw_seq_recovery.append(raw_seq_recovery_init[i])
        raw_seq_recovery_plot.append(raw_seq_recovery_init[i]*100)
    pen = pg.mkPen(color = 'g', width = 3)
    graph2.clear()
    graph2.plot(a,raw_seq_recovery_plot, pen = pen, symbol = '+')

def on_toggle():
    global flag
    global dirpath
    if flag == 0:
        flag =1
        filepath = dirpath+'/map1.html'
        toggle.setText("Heat Map")
        x.setUrl(QUrl.fromLocalFile(filepath))
    elif flag == 1:
        flag =2
        filepath = dirpath+'/map3.html'
        toggle.setText("Hospital Near Me")
        x.setUrl(QUrl.fromLocalFile(filepath))
    elif flag == 2:
        flag = 0
        filepath = dirpath+'/map0.html'
        toggle.setText("Cases Map")
        x.setUrl(QUrl.fromLocalFile(filepath))
    else:
        pass
    x.show()
    
app = QApplication(sys.argv)
w = QMainWindow()
w.setWindowTitle('COVID-19 UPDATES')
w.setGeometry(10,10,1200,800)
x = QWebView(w)
x.setUrl(QUrl.fromLocalFile(filepath))
x.move(20,20)
x.resize(700,600)
x.show()
toggle = QPushButton("Cases Map", w)
toggle.resize(300,50)
toggle.move(210, 650)
toggle.clicked.connect(on_toggle)
toggle.setStyleSheet("background-color: yellow")

lay = QPushButton("Predict", w)
lay.resize(200,50)
lay.move(850,330)
lay.clicked.connect(on_predict_cases)
lay.setStyleSheet("background-color: green")
reset = QPushButton("Reset", w)
reset.resize(200,50)
reset.move(1100,330)
reset.clicked.connect(on_reset)
reset.setStyleSheet("background-color: red")
graph1 = pg.PlotWidget(w)
a = x_axis_cases
b= raw_seq_cases
raw_seq_cases_plot = []
for i in range(len(raw_seq_cases)):
    raw_seq_cases_plot.append(2000*raw_seq_cases[i])   
pen = pg.mkPen(color = 'g', width = 3)
graph1.plot(a,raw_seq_cases_plot, pen = pen, symbol = '+')
#graph1.plot(a,b, pen = pen)
graph1.setBackground('w')
graph1.setLabel('left', "<span style=\"color:red;font-size:20px\">Cases in 1000s</span>")
graph1.setLabel('bottom', "<span style=\"color:red;font-size:20px\">Days</span>")
graph1.setGeometry(800,10, 500, 300)
graph2 = pg.PlotWidget(w)
a = x_axis_recovery
b= raw_seq_recovery
raw_seq_recovery_plot = []
for i in range(len(raw_seq_recovery)):
    raw_seq_recovery_plot.append(raw_seq_recovery[i]*100)
pen = pg.mkPen(color = 'g', width = 3)
graph2.plot(a,raw_seq_recovery_plot, pen = pen, symbol = '+')
graph2.setLabel('left', "<span style=\"color:red;font-size:20px\">Recovery Rate</span>")
graph2.setLabel('bottom', "<span style=\"color:red;font-size:20px\">Days</span>")
graph2.setBackground('w')
graph2.setGeometry(800,400, 500, 300)
w.show()
app.exec_()
