from numpy import array
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import random

def split_seq(seq, n_steps):
    x,y = list(), list()
    for i in range(len(seq)):
        end_ix = i + n_steps
        if end_ix > (len(seq)-1):
            break
        seqx, seqy = seq[i:end_ix], seq[end_ix]
        x.append(seqx)
        y.append(seqy)
    return x,y
#######################################################################################
### THIS IS FOR THE RECOVERY RATE#######
########################################################################################
raw_seq = [58.20,60.15,64.83,67.49,71.00,74.21,76.75,78.50,79.89,81.16,82.60,83.12,
           84.30,85.11,86.08,87.16,87.87,88.50,89.85,89.85,90.20,90.52,91.05,91.58,
           92.14,92.74,93.10,93.43,93.67,93.92,94.09,94.22,94.26,94.36,94.21,94.09,
           93.94,93.65,93.39,93.04,92.87,92.24,91.75,91.19,90.51,89.78,88.94,88.01,
           87.07, 86.07, 85.21, 84.29, 83.79, 82.95, 82.14, 81.62, 81.41, 80.81, 80.44,
           80.44, 79.95]

n_steps = 12
x_axis = []
for i in range(len(raw_seq)):
    raw_seq[i] = raw_seq[i]/ 100
for i in range(len(raw_seq)):
    x_axis.append(i)

new_x_axis = len(raw_seq)-1
plt.plot(x_axis, raw_seq)   

x,y = split_seq(raw_seq, n_steps)
x = np.array(x)
y = np.array(y)
#print(x)
#print(y)
x_train = []
y_train = []
x_val = []
y_val = []
for i in range(len(x)):
    rand = random.random()
    if rand < 0.2:
        x_val.append(x[i])
        y_val.append(y[i])
    else:
        x_train.append(x[i])
        y_train.append(y[i])

x_val = np.array(x_val)
y_val = np.array(y_val)
x_train = np.array(x_train)
y_train = np.array(y_train)


##    print(x[i], y[i])
n_features = 1   
model = Sequential()
model.add(LSTM(50, activation = 'relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss = 'mse')
print("Model built")
n_features = 1
x = x.reshape((x.shape[0],x.shape[1], n_features))

x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], n_features))
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))

#model.fit(x_train, y_train, epochs = 1000, verbose = 2)
model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 1000)
model.save("recovery.h5")
#model = load_model('recovery.h5')
x_axis_values = [new_x_axis]
y_axis_values = [raw_seq[new_x_axis]]
for i in range(35): 
    x_input = raw_seq[(len(raw_seq)-n_steps):]
    print("ARRAY:", x_input)
    x_input = np.array(x_input)
    x_input = x_input.reshape((1,n_steps, n_features))
    new_y = np.array(model.predict(x_input, verbose = 0))
    print(new_y[0][0])
    raw_seq.append(new_y[0][0])
    x_input = raw_seq[(len(raw_seq)-n_steps):]
    print("ARRAY:", x_input)
    #print("Len of raw seq:", len(raw_seq))
    new_x_axis += 1
    x_axis_values.append(new_x_axis)
    y_axis_values.append(new_y[0][0])
    
    #plt.plot(new_x_axis, new_y)
    print(new_y)

plt.plot(x_axis_values, y_axis_values)
plt.ylim(0, 1)
plt.show()
###########################################################################
## THIS IS FOR THE TOTAL CASES
###########################################################################
raw_seq = [580, 845, 1317, 2015, 2800, 4581, 6058, 7813,
     9823, 11950, 14553, 17391,20630, 24545, 28266, 31439, 34876,
     37552, 40553,43099, 45134, 59287, 64438, 67100, 69197, 71329,
     73332, 75184, 75700,76677, 77673, 78651, 79205, 80087, 80828, 
     81820, 83112,84615, 86604, 88585, 90443, 93016, 95314, 98425,
     102050, 106099, 109991,114381, 118948, 126948, 134576, 145483,
     156653, 169593,182490, 198238, 218822, 244933, 275597, 305036,
    337000, 378000, 422000, 471000, 531000, 596000, 663000, 723000,
    784000, 858000, 935000, 1010000]

n_steps = 12
x_axis = []
for i in range(len(raw_seq)):
    raw_seq[i] = raw_seq[i]/ 2000000
for i in range(len(raw_seq)):
    x_axis.append(i)

new_x_axis = len(raw_seq)-1
plt.plot(x_axis, raw_seq)   

x,y = split_seq(raw_seq, n_steps)
x = np.array(x)
y = np.array(y)
#print(x)
#print(y)
x_train = []
y_train = []
x_val = []
y_val = []
for i in range(len(x)):
    rand = random.random()
    if rand < 0.3:
        x_val.append(x[i])
        y_val.append(y[i])
    else:
        x_train.append(x[i])
        y_train.append(y[i])

x_val = np.array(x_val)
y_val = np.array(y_val)
x_train = np.array(x_train)
y_train = np.array(y_train)


##    print(x[i], y[i])
n_features = 1   
model = Sequential()
model.add(LSTM(50, activation = 'relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss = 'mse')
print("Model built")
n_features = 1
x = x.reshape((x.shape[0],x.shape[1], n_features))

x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], n_features))
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))

model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 1000)
model.save("cases.h5")
#model = load_model('cases.h5')
x_axis_values = [new_x_axis]
y_axis_values = [raw_seq[new_x_axis]]
for i in range(10): 
    x_input = raw_seq[(len(raw_seq)-n_steps):]
    x_input = np.array(x_input)
    x_input = x_input.reshape((1,n_steps, n_features))
    new_y = np.array(model.predict(x_input, verbose = 0))
    print(new_y[0][0])
    raw_seq.append(new_y[0][0])
    #print("Len of raw seq:", len(raw_seq))
    new_x_axis += 1
    x_axis_values.append(new_x_axis)
    y_axis_values.append(new_y[0][0])
    
    #plt.plot(new_x_axis, new_y)
    print(new_y)

plt.plot(x_axis_values, y_axis_values)
plt.show()

