# LSTM Abu Dhabi Weather Predictor


## Overview

This is a TensorFlow and Pytorch-based implementation of a two-layered LSTM network that trains on Abu Dhabi weather data from OpenMeteo API from the last 30 days, to then make a prediction about the next hour's statistics. This program also benchmarks the two models by comparing MSE and MAE, and their respective performances on each statistic. Statistics are: 

* Temperature(2m)
* Relative humidity(2m)
* Apparent temperature
* Precipitation probability
* Cloud cover(high)
* Wind direction (80m)

---

## Contents

- `PytorchModel.py`: Executable Pytorch LSTM that trains on the last 30 days of weather data and visually with MatPlotLib shows its prediction. 
- `TensorFlowModel.py`: Executable TensorFlow LSTM that trains on the last 30 days of weather data and visually with MatPlotLib shows its prediction. 
- `Benchmark.py`: Executable file that creates both models and then displays their individual performances, their performances on the same graph, then prints their MSE and MAE.
- `WeatherData.py`: API code that retrieves hourly weather statistics from the last 30 days in Abu Dhabi. 

---

## How to use

Execute any file of the three executable files, depending on if you want to train the PyTorch model, the TensorFlow model, or benchmark them respectively.

---

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- requests
- requests-cache
- retry-requests
- TensorFlow 2.x
- PyTorch (torch, torchvision)

---

## Relevance

This project was developed as part of me wanting to learn how to use proper AI libraries, after having built a FFNN manually. I wanted to both understand the architecture of these models, AND be able to build them the industry-standard way, e.g. with libraries like PyTorch and TensorFlow. 

---