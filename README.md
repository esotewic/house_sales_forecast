# Forecasting Real Estate Market with Time Series Modeling

## Data
The dataset was pulled from the MLS (Multiple Listing Services). There are about 26,000 transactions that span from Jan 2016 to Sept 2019. In a timeframe of slightly less than 4 years, there has been $40~ bn in total transactions.

## Objective
This project aims to utilize machine learning on real estate data to accurately predict the future weekly home sale count and weekly average sale price in West Los Angeles Areas. The housing market in greater LA and its make-up of unique neighborhoods has always been a topic of great discussion among brokers and lenders. The potential applications of predicting future outcomes will allow us to identify potential investment opportunities. My goal is to create forecast model to predict prices & demand in each of these unique markets. Real estate firms or lenders could potentially use this information to fill voids in the market.

<img src="https://github.com/esotewic/house_sales_forecast/blob/master/pictures/LA_volume.png">

## Overview
Goal:
Determine if a certain neighborhood is a good place to invest based on the forecasted median sale price and volume of sales (demand).
Result:
All three neighborhoods have potential investment value because of upward trend of sale prices. Only one neighborhood (Mid-Wilshire) exhibited upward trend in demand volume.

<img src="https://github.com/esotewic/house_sales_forecast/blob/master/pictures/count_forecast.png">

## Applications
- Investors looking for Real Estate opportunities
- Real estate brokerage firms can use this to pick out potential new farm targets. Zillow has a similar feature but does is limited to their drop down selections. I would like to take it one step further and be able to grab specific property types in specific neighborhoods.
- Mortgage banks can use data for potential new purchase applications or refinances

### Dependencies
- Python 3.7.2  
- fbprophet 0.4.post2
- pandas 0.24.1
- NumPy 1.16.2
- scikit-learn 0.20.3
- Matplotlib 3.0.3

### Results
***Overview***
Testing:
Santa Monica/Silverlake/Mid-Wilshire
Model Price MAPE’s: 1.9%/0.6%/0.9%
Baseline Price MAPE’s: 2.1%/1.0%/1.1%
Model Demand MAPE’s: 2.6%/5.3%/5.1%
Baseline Demand MAPE’s: 4.7%/7.7%/6.7%
<img src="https://github.com/esotewic/house_sales_forecast/blob/master/pictures/sm_volume.png">

## Process
#### 1. Exploratory Data Analysis
  - Property Type seemed to be the best categorical feature to help the model "learn".
    - E.g. LivingArea, Bedrooms, Bathrooms
  - Different PostalCodes labeled definitely exhibited some similarities.
  - Data in greater Los Angeles area does not share same qualities as other cities in America. I wanted to test the uniqueness of the housing infrastructure of the area.
  - Refer to Housing Price Model for data cleaning

#### 2. Defined Baseline using naive forecasting
  - Compared using the same timeframe from previous year

#### 3. Engineered Generalized Additive Model
 - Use Facebook Prophet for forecasting

#### 4. Slice data into different areas of focus
  - Type and Location features
  - Easy to look into other feature but started with simplest

#### 5. Measured Model Outcomes and examined results using MAPE metric
  - Comparing different features and target sizes impact MAPE


## Prophet
Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. The method was first open sourced by Facebook in late 2017, and is available in Python and R.
### Model
Prophet makes use of a decomposable time series model with three main model components: trend, seasonality, and holidays.
`y(t) = g(t) + s(t) + h(t) + e(t)`
where:
- g(t)
    - *trend* models non-periodic changes (i.e. growth over time)
- s(t)
    - *seasonality* presents periodic changes (i.e. weekly, monthly, yearly)
- h(t)
    - ties in effects of *holidays* (on potentially irregular schedules ≥ 1 day(s))
- e(t)
    - covers idiosyncratic changes not accommodated by the model
- For more on the equation behind the procedure, check out [The Math of Prophet](https://medium.com/future-vision/the-math-of-prophet-46864fa9c55a) [10 min]
- To learn more about how to use Prophet, see [Intro to Facebook Prophet](https://medium.com/future-vision/intro-to-prophet-9d5b1cbd674e) [9 min]
### Specifications
As the question at hand relied weekly data points, `Prophet`, was set to exclude daily  seasonality while staying alert when identifying year-to-year trends and shifts in those trends over time. This was achieved;  
```
# use fbprophet to make Prophet model
master_model=Prophet(
    growth='linear',
    n_changepoints=4,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='additive',
    ).add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=5
    ).add_seasonality(
        name='yearly',
        period=365.25,
        fourier_order=20
    ).add_seasonality(
        name='quarterly',
        period=365.25/4,
        fourier_order=5,
        prior_scale=15)
```
## Conclusion
Silverlake, Mid-Wilshire, and Santa Monica all have an upward trend in forecast regarding sale prices. However, Mid-Wilshire is the only neighborhood exhibiting an increasing trend in sales volume in prior and forecasted. As a lender or brokerage this could still be a viable opportunity for increased profitability in commision or margins but as a agent farming I would be hesitant to enter other markets.

<img src="https://github.com/esotewic/house_sales_forecast/blob/master/pictures/midwilshire_forecasted_trend.png">
