# The Impact of BARJAM (JCPOA) Cancellation on the Iranian Dollar Exchange Rate

## Project Overview

This project aims to analyze and visualize the impact of the cancellation of the Joint Comprehensive Plan of Action (JCPOA), commonly known as BARJAM, on the exchange rate between the Iranian Rial (IRR) and the US Dollar (USD). The project leverages historical exchange rate data, various statistical and machine learning models, and visualizations to provide insights into the currency trends and potential scenarios had BARJAM not been canceled.

This dataset contains daily exchange rates of the Iranian Rial (IRR) against the US Dollar (USD). The data was collected using a web crawler and includes columns for opening, low, high, and closing rates, as well as the average rate for each day.
I personally created this dataset using a web crawler to gather data from various financial sources.

### Data Collection

The dataset was created through a web crawling process undertaken by myself, ensuring comprehensive and accurate data extraction from various financial sources.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Visualizations](#visualizations)
6. [Models and Analysis](#models-and-analysis)
7. [How to use website?](#website)

## Dataset

The dataset used in this project contains historical exchange rates of the Iranian Rial against the US Dollar. The dataset includes columns such as Date, Open, Low, High, Close, and Mean exchange rates.

- **Source:** [Dataset Link](https://www.kaggle.com/)
- **Format:** Excel file (`Dollor_Rate_Dataset.xlsx`)

## Installation

To run this project, you need to have Python installed along with the following libraries:

- numpy
- pandas
- matplotlib
- seaborn
- warnings
- prophet
- scikit-learn
- streamlit
- plotly
- statsmodels

You can install these libraries using `pip`:

```bash
pip install numpy pandas matplotlib seaborn warnings prophet scikit-learn streamlit plotly statsmodels
```

## Visualizations

![1](https://github.com/user-attachments/assets/f422753e-b428-4c84-b370-37bacdb13153)
![2](https://github.com/user-attachments/assets/cc738d6b-4c28-43c7-9d9b-023d0d605f66)
![3](https://github.com/user-attachments/assets/afb05edd-5f20-4698-86f3-e5899d5d5d9f)

## Models-and-Analysis

Forecasted by Mata prophet and other LSTM networks & regressors.
Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data.

Prophet is open source software released by Facebook’s Core Data Science team. It is available for download on CRAN and PyPI.

```bash
pip install prophet
```

## Website

In order to use website you must follow the instruction of streamlit library to deploy (webserver.py) on your localhost.

```bash
pip install numpy pandas matplotlib seaborn warnings prophet scikit-learn streamlit plotly statsmodels

streamlit run https://github.com/Matin-Rasoulzad/The-Impact-of-BARJAM--JCPOA--Cancellation-on-the-Iranian-Dollar-Exchange-Rate/blob/main/webserver.py
```
![image](https://github.com/user-attachments/assets/0cd67d9c-ca30-446a-b816-0309b4efaff9)

Otherwise please click the link below:

https://matinrasoulzad-barjam.streamlit.app
