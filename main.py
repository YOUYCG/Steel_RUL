# Importing libraries
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns 
import os 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from joblib import dump ,load
plt.style.use('ggplot')
fontdict = {
    'family': 'sans-serif',
    'color':  'black',
    'weight': 'normal',
    'size': 14,
}

# Loading data
df_1=pd.read_csv('gt_data.csv')

# Display information about the dataframe
df_1.info()

# Display the shape of the dataframe
df_1.shape

# Check for null values
df_1.isnull().sum()

# Display column names
df_1.columns

# Create a copy of the dataframe with more descriptive column names
df_3=df_1.copy()
df_3.columns=['Sl. No.','Normalizing Temp','Through Hardening Temp','Through Hardening Time','Cooling Rate for Through Hardening','Carburization Temp',
             'Carburization Time','Diffusion Temp','Diffusion time','Quenching Media Temp','Tempering Temp','Tempering Time','Cooling Rate for Tempering',
             'C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Cu', 'Mo','Reduction Ratio',
              'Area Proportion of Inclusions Deformed by Plastic Work','Area Proportion of Inclusions Occurring in Discontinuous Array',
              'Area Proportion of Isolated Inclusions','Fatigue Strength (10^7 Cycles)']

# Display descriptive statistics for all columns except the first one
df_3.iloc[:,1:].describe().T

# Histogram of columns
df_3.iloc[:,1:14].hist(figsize=(25,25))
plt.suptitle('histogram of columns',fontsize=22)
plt.show()

# Correlation between diffrent features and Fatigue Strength
# green columns indicate positive correlation (Direct propotionalty)
# red columns indicate negative correlation (Inverse propotionality)
correlation = df_3.drop(columns=['Sl. No.','Fatigue Strength (10^7 Cycles)']).corrwith(df_3['Fatigue Strength (10^7 Cycles)'])
correlation.sort_values(inplace=True)
plt.figure(figsize=(14,7))
plt.bar(correlation.index, correlation, color=['green' if val >= 0 else 'red' for val in correlation])
plt.xlabel('Features')
plt.ylabel('Correlation')
plt.title('Correlation with Fatigue Strength')

plt.xticks(rotation=45, ha='right',fontsize=14)
plt.show()

# Data of top 10 fatigue strength sampels
df_3.sort_values('Fatigue Strength (10^7 Cycles)',ascending=False).head(10).iloc[:,:14]

df_3.sort_values('Fatigue Strength (10^7 Cycles)',ascending=False).head(10).iloc[:,14:]

# Creating Regression Model

# Spiltting data into train and test
x=df_3.drop(columns=['Sl. No.','Fatigue Strength (10^7 Cycles)'])
y=df_3['Fatigue Strength (10^7 Cycles)']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)

# Creating a Random forest regression model
forest_reg=RandomForestRegressor(n_estimators=100, max_depth=100)
forest_reg.fit(x_train,y_train)

y_pred_1=forest_reg.predict(x_test)
rmse_forest=np.sqrt(mean_squared_error(y_test,y_pred_1))
r2_forest=r2_score(y_test,y_pred_1)
scores = cross_val_score(forest_reg, x, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)

print('RMSE for Random Forest:',rmse_scores.mean()) 
print('R squared for Random Forest:',r2_forest) 

# Feature that has affects Fatigue Strength most
labels=x_train.columns
importanes=forest_reg.feature_importances_
importance_dict=dict(zip(labels,importanes))
importance_series=pd.Series(importance_dict)
importance_series=importance_series.sort_values(ascending=True)
plt.figure(figsize=(12,9))
importance_series.plot(kind='barh')
plt.title('Features that affect Fatigue Strength most',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Effect',fontdict=fontdict)
plt.show()

# Creating a Artifitial Neural Network for regression
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
scaler_1 = MinMaxScaler()

X_train_scaled = scaler.fit_transform(x_train.values)
X_test_scaled = scaler.transform(x_test.values)
y_train_scaled=scaler_1.fit_transform(y_train.values.reshape(-1,1))
y_test_scaled=scaler_1.transform(y_test.values.reshape(-1,1))

model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1)  # Output layer with 1 neuron for regression
])
from keras.callbacks  import EarlyStopping 
early_stopping=EarlyStopping(restore_best_weights=True,monitor='val_mse',mode='min',patience=40,verbose=1)
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mse'])
history = model.fit(X_train_scaled, y_train_scaled, epochs=200, batch_size=32, validation_split=0.2,callbacks=[early_stopping])

def plot_loss_and_mse_curves(history):
    """
    Plots the loss and MSE curves from a model's training history.

    Parameters:
    - history: The history object returned by the fit method of a Keras model.
               It should contain the keys 'loss', 'val_loss', 'mse', and 'val_mse'.
    """
    # Get the values from the history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mse = history.history.get('mse')
    val_mse = history.history.get('val_mse')
    
    epochs = range(1, len(loss) + 1)

    # Plot the loss values
    plt.figure(figsize=(12, 6))

#     plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot the MSE values if they exist
#     if mse and val_mse:
#         plt.subplot(1, 2, 2)
#         plt.plot(epochs, mse, 'b', label='Training MSE')
#         plt.plot(epochs, val_mse, 'r', label='Validation MSE')
#         plt.title('Training and Validation MSE')
#         plt.xlabel('Epochs')
#         plt.ylabel('Mean Squared Error')
#         plt.legend()

#     plt.tight_layout()
#     plt.show()

plot_loss_and_mse_curves(history)

y_pred_2=model.predict(X_test_scaled)
print('R squared for Artificial neural network: ',r2_score(scaler_1.inverse_transform(y_test_scaled),scaler_1.inverse_transform(y_pred_2)))
print('RMSE for Artificial neural network: ',np.sqrt(mean_squared_error(scaler_1.inverse_transform(y_test_scaled),scaler_1.inverse_transform(y_pred_2))))

# R squared score is higher for Random forest and rmse is lower for it than Artificial neural network so we will save it to use it later for streamlit app
dump(forest_reg,'forest_reg.joblib')

import pickle 
pickle.dump(model,open('ann_model.pkl','wb'))

model_1=pickle.load(open('ann_model.pkl','rb'))
model_1 

y_pred_3=model_1.predict(X_test_scaled)
print('R squared for Artificial neural network: ',r2_score(scaler_1.inverse_transform(y_test_scaled),scaler_1.inverse_transform(y_pred_3)))
print('RMSE for Artificial neural network: ',np.sqrt(mean_squared_error(scaler_1.inverse_transform(y_test_scaled),scaler_1.inverse_transform(y_pred_3))))
