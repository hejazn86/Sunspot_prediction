# Sunspot_prediction

## Purpose of the project:
* Building a model to predict a time series data(sunspot data-set from Kaggle)
* To invastigate the performance of different kinds of layers in time_series prediction process (Dense, RNN, LSTM , Convolutional)
* To invastigate the impact of different learning rate,diffrent kind of loss function, and different window_size on the prediction process

## Preparing the Data
we have used the **Sunspots dataset** from *Kaggle*, a *CSV* file contains avrage monthly amounts of measured sunspots and the dates on which they are measured. The sunspots have *seasonal cycles* approximately every 11 years, and we are going to try to predict these seasonal cycles by our model.

### Download the data to the drive
Firstly, I download the data from *Kaggle* and upload it to *Google drive*, then I connected the drive to  the *Colab* notebook by mounting the drive using the code below:
```
from google.colab import drive
drive.mount('/drive')
```

### Upload and Read the data
The next step is to upload the data from the drive to the notebook and sort it. This task is done by using `csv` and `Numpy` libraries. We first import these two libraries, and create empty lists `[]` both for **time_steps** ans **sunspots**, then use the `open` and the `csv.reader` functions to open and read the file of data respectively. After that a `for` loop is used to iterate over the file and add the data to *time_steps* and *sunspots* lists, but before that we use `next` function to skip the header of the file. The first and the third columns in the file represent *time steps* and *average amounts of sunspots*. Eventually we convert the lists to numpy arrays for convenience and memory optimization:

```
series = np.array(sunspots)
time = np.array(time_steps) 
```

* `We use matplotlib.pyplot` library to show the data and how it behave over the time steps.

### Split the data into Train and Test datasets
In this step, we will split the data in two datasets: *Train dataset* and *Test dataset*
We use the time steps to split the data. The data contains *3252* time steps and Sunspot amounts, so initially we split the data from the *20000th* time step, setting the first *20000th* average amounts of sunspots as the *tarin set* and the rest as *test set*, as the following code shows:

```
split_time = 3500
train_time =time[:split_time]
x_train = series[:split_time]
test_time = time[split_time:]
x_test = series[split_time:]
```

### Preparing the dataset
at the last step of preparing the data, we create a function to process the data based on the desired *window size*, *batch size* and *suffle_buffer size* to fit the model:

```
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset
```

The above code takes 4 arguments: 
* **series**: The data (In this case only the average amount of sunspots)
* **the window_size**: A number of time steps used as a base to be processed and analyzed to predict the next item (e.g. using the first 6 average amounts of sunspots to predict the 7th amount, so the 7th amount will be set as a label for them) 
* **batch_size**: Size of batchs to which the data divided in order to be fed to the model at each iteration
* **shuffle_buffer**: The number of the data from the dataset from which the dataset will be sampled








