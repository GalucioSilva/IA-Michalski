import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, KFold
import keras 
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate
from keras.losses import binary_crossentropy
from keras.utils  import plot_model
from keras_radam import RAdam
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

seed_value= 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


# path to trains-transformed.csv
path = '/home/galucio/Documentos/TrabFinalIA/trains-transformed.csv'
str_att = {
  'length': ['short', 'long'],
  'shape': ['closedrect', 'dblopnrect', 'ellipse', 'engine', 'hexagon',
          'jaggedtop', 'openrect', 'opentrap', 'slopetop', 'ushaped'],
  'load_shape': ['circlelod', 'hexagonlod', 'rectanglod', 'trianglod'],
  'Class_attribute': ['west','east']
}

def read_data(path=path):
  df = pd.read_csv(path, ',')

  for k in df:
    for att in str_att:
      if k.startswith(att):
        for i,val in enumerate(df[k]):
          if val in str_att[att]:
            df[k][i] = str_att[att].index(val)

  df.replace("\\0", 0, inplace=True)
  df.replace("None", -1, inplace=True)

  return df

df = read_data()
print(df)


# get data
df = read_data()
Y = np.array(df.pop('Class_attribute'))
X = np.array(df)

# define model
def model_v1():
  model = Sequential([
    Dense(9, activation='relu', input_shape=(len(df.keys()),)),
    Dense(1, activation='sigmoid'),
  ])
  model.compile(
    loss = binary_crossentropy, 
    optimizer = RAdam(),
    metrics = ['mse', 'binary_accuracy']
  ) 
  return model

es = keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=1000, verbose=0)

model = model_v1()
model.summary()

plot_model(model, show_shapes=True, show_layer_names=False)

# train
loo = LeaveOneOut()
hists = dict()
q1_names =  [
  'Train', 
  'Output of flat network', 
  ' Desired output', 
  'Class'
]
q1 = {name: [] for name in q1_names}

for i, (train, test) in tqdm.tqdm(enumerate(loo.split(X)), total=10):
  x_train = X[train]
  y_trian = Y[train]
  x_test  = X[test]
  y_test  = Y[test] 

  model = model_v1()
  hist = model.fit(
    x_train,
    y_trian,
    validation_data=[x_test, y_test],
    epochs=10000,
    verbose=0,
    # callbacks=[es]
  )
  hists[test[0]] = hist

  q1['Train'].append(i)
  q1['Output of flat network'].append(model.predict(x_test)[0][0])
  q1[' Desired output'].append(y_test[0])
  q1['Class'].append(str_att['Class_attribute'][int(y_test)])

for i,history in enumerate(hists.values()):
  plt.plot(history.history['mean_squared_error'], c='tab:blue')
  plt.plot(history.history['val_mean_squared_error'], c='tab:orange')
  
plt.title('model mean_squared_error')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

for i,history in enumerate(hists.values()):
  plt.plot(history.history['binary_accuracy'], c='tab:blue')
  plt.plot(history.history['val_binary_accuracy'], c='tab:orange')
  
plt.title('model binary_accuracy')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

t1 = pd.DataFrame.from_dict(q1).round(2)
print(t1)

model_json = east.to_json()
with open("model.json", "w") as json_file:
  json_file.write(model_json)
  #serialize weights to HDF5
east.save_weights("model.h5")
print("Saved model to disk")