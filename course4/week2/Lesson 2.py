import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# generate a random series
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)  
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)

# split the data
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

# setup window and batch sizes
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

# function to return a windowed dataset
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

# get a windowed dataset based on the training data
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

# setup the NN
l0 = tf.keras.layers.Dense(1, input_shape=(window_size,))
model = tf.keras.models.Sequential([l0])

# compile the model
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))

# fit the data on the dataset
# dataset contains x and y
model.fit(dataset, epochs=100, verbose=1)

# get the 20 (window size) layer weights
# 19 of them will be weights, and the 20th is the b (y intercept) value
# new value = Wt0 * X0 + Wt1 * X1 + ... + Wt19 * X19 + b
print("Layer weights {}".format(l0.get_weights()))

# prediction
forecast = []

for time in range(0, len(series) - window_size):
    # given window size amount of elements, predict the window + 1 value
    # the np.newaxis just converts [1 2 3 ...] to [[ 1 2 3 ...]]
    forecast.append(model.predict(series[time: time + window_size][np.newaxis]))

# only take the validation set in the forecasting
forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]


plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)
plot_series(time_valid, results)
plt.show()

# measure the accuracy
# compare the valid (series after the split time) with the results we predicted
print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())