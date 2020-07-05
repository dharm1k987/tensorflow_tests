'''
Time series

A time series is a series of data points indexed in time order
Usually the time sequence is at successive equally spaced points

For instance, stock prices over a time interval, birth rates per year, etc

Timeseries have the following terms:

Trend - going up or now
Seasonality - up for a certain time, down for the other (local peeks and troughs)
Noise - random values
Autocorrelation - no trend and no seasonality, but the entire series isn't random

We will use fixed partioning to measure our accuracy:
There is a large training period, and a smaller validation period
The period here is just the combination of time and values over an interval
The period must be successive however, unlike previous models where we just chose
random values from the corpus

Evaluating metrics:
errors = forecasts - actual
mse = np.square(errors).mean()
mae = np.abs(errors).mean()

Forecast is what we predict and actual is the actual data in the validation set

We can forecast based on purely statistical methods instead of deep learning:
1) Naive forecast
    - just take time t - 1 value as time t's value

2) Moving average
    - create a smooth curve that goes over the data and predict based on that
    - this is not too good at inferring about trends however

3) Moving average with differencing
    - subtract time t with time t - interval to remove any trends
    - find a moving average on that
    - add back time interval to this to get a model for the current time
    - run moving average on this once more
'''

