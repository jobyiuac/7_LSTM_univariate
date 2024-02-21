
# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt

# define dataset
X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
print('X before reshaping: \n\n', X)
print('\nshape is: ', X.shape)
y = array([40, 50, 60, 70])

# reshape from [samples, timesteps] into [samples, timesteps, features] that is 3*1 matrix
X = X.reshape((X.shape[0], X.shape[1], 1))

print('\nX after reshaping: \n\n', X)
print('\n shape becomes: ', X.shape)

# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=1000, verbose=0)

# demonstrate prediction
x_input = array([50, 60, 70])
x_input = x_input.reshape((1, 3, 1))
print('\n new input: \n\n',x_input)
yhat = model.predict(x_input, verbose=0)

print('\n predicted value: ',yhat)


