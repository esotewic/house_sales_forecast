# Forecasting House Sales & Price using Facebook Prophet
- This is a continuation of analysis on my previous machine learning project to predict home prices using data from the MLS (multiple listing services)
- [Home Price Prediction](https://github.com/esotewic/ultimate_housing_model)
## Premise
This project aims to utilize machine learning on real estate data to accurately predict the future weekly home sale count and weekly average sale price in West Los Angeles Areas.
## Applications
- Real estate brokerage firms can use this to pick out potential new farm targets
- Mortgage banks can use data for potential new purchase applications or refinances
### Presentation
- [One Pager]
- [Resume]
### Dependencies
- Python 3.7.2  
- fbprophet 0.4.post2
- pandas 0.24.1
- NumPy 1.16.2
- scikit-learn 0.20.3
- Matplotlib 3.0.3
### Results
***Overview***
- The model preformed with an average 8.96% cross validated Mean Absolute Percentage Error
    - Calculated by comparing the final six months of the training's actual values with the Model's predictions for that year
- Predictions on different property types and cities within Los Angeles area returned different results because of target sizes
    - Averaged 11.08% for Single Family Homes and 13.33% MAPE for Condos
    - When examining different cities (Santa Monica, Beverly Hills, Silverlake) MAPE's were drastically different
        - Averaged 27.78%, 15.72%, and 39.39% MAPE (respectively)

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
place_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15,
                                  daily_seasonality=False,
                                  weekly_seasonality=False,
                                  yearly_seasonality=True,
                                  n_changepoints=10)
```
## Results  
- The model preformed with an average 8.96% cross validated Mean Absolute Percentage Error
    - Calculated by comparing the final six months of the training's actual values with the Model's predictions for that year
- Predictions on different property types and cities within Los Angeles area returned different results because of target sizes
    - Averaged 11.08% for Single Family Homes and 13.33% MAPE for Condos
    - When examining different cities (Santa Monica, Beverly Hills, Silverlake) MAPE's were drastically different
        - Averaged 27.78%, 15.72%, and 39.39% MAPE (respectively)
