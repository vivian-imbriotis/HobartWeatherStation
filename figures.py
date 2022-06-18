# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:05:39 2022

@author: vivia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from datetime import datetime
import os

import statsmodels.formula.api as smf


sns.set_style("darkgrid")

df = pd.read_csv("Ellerslie_rd_daily_temp/IDCJAC0010_094029_1800_Data.csv")

#Turn year, month, and day fields into datetime objets

years = [int(year) for year in df.Year]
months= [int(month) for month in df.Month]
days  = [int(day) for day in df.Day]

datetimes = [datetime(y,m,d) for y,m,d in zip(years,months,days)]

#make a pandas index out of those datetimes
datetimes = pd.DatetimeIndex(datetimes)
df.index = datetimes

#Now that we've got the index we can drop the unneeded columns
df["temperature"] = df["Maximum temperature (Degree C)"]
df = df.drop(["Year","Month","Day", "Product code", 
              "Bureau of Meteorology station number",
              "Maximum temperature (Degree C)",
              'Days of accumulation of maximum temperature'], 
             axis = "columns")


#Storing the figure objects in a directory
if not os.path.isdir("figures"):
    os.mkdir("figures")

#First figure - Just a simple scatter plot
fig,ax = plt.subplots(figsize=[9.0,4.8], tight_layout=True)
dot_artist, = ax.plot(df.temperature,'o', alpha = 0.4, ms = 1.5)
ax.set_ylabel("Maximum temperature (Degree C)")
ax.set_xlabel("Year")
ax.set_title("Tasmanian Temperatures")
fig.savefig("figures/raw_temps_dots")

#Turn the scatterplot into a line plot
dot_artist.remove()
ax.plot(df.temperature,linewidth = 0.5, color='k', alpha = 0.9)
fig.savefig("figures/raw_temps_line")


#Time to build our model. Our exogenous terms are going to be 
#the time elapsed since data recording began in days, as well
#as a single sine basis expansion function to represent the 
#season. We expect the seasonal temperature to peak in feburary
#so I just set feburary to be the antinode.
df["time_elapsed"] = (datetimes - datetimes[0]).days.to_series().values

df["seasonality"] = np.cos(2*np.pi*(datetimes.day_of_year - 31) / 365)


#Now just use OLS to fit that to the teperature
model = smf.ols(formula = "temperature ~ time_elapsed + seasonality", data=df)
result = model.fit()
preds = result.get_prediction().summary_frame()
parameters = result.params
conf_int = result.conf_int()

#Now of couse we've violated heaps of assumptions of linear regression, ay
#For example, variability is higher in the summer months so the data isn't
#exactly homoskedastic. This means we can't exactly believe our confidence
#intervals / t-tests for significance. In lieu of a more sophisticated 
#approach we could bootstrap some CIs if we wished.


fig = plt.figure(figsize=[14,6], tight_layout=True)
gs = GridSpec(2, 4, figure = fig)
ax = fig.add_subplot(gs[:,:-1])
lin_ax = fig.add_subplot(gs[0,-1])
season_ax = fig.add_subplot(gs[1,-1])
ax.plot(df.temperature, linewidth = 0.5, label = "Observed temperature")
ax.fill_between(preds.index, preds.obs_ci_lower, preds.obs_ci_upper, 
                 color = sns.color_palette()[1],
                 where = np.equal(preds.index.to_series().diff().dt.days,1),
                 label = "Model prediction (95% certainty)")
ax.legend()


lin_ax.fill_between(datetimes, df.time_elapsed * conf_int[0].time_elapsed + conf_int[0].Intercept,
                    df.time_elapsed * conf_int[1].time_elapsed + conf_int[1].Intercept,
                    color = sns.color_palette()[1], alpha = 0.5, label="Average temperature (with 95% CI)")

lin_ax.plot(datetimes, np.ones(datetimes.shape)*df.temperature.mean(), 
       linestyle="--", color = 'k',
       label = "Null Hypothesis")

lin_ax.plot(datetimes,df.time_elapsed * parameters.time_elapsed + parameters.Intercept,
            color = sns.color_palette()[1])

lin_ax.set_ylabel("Mean max daily Temp (Degree C)")
lin_ax.set_xlabel("Year")
lin_ax.set_title("Linear trend in Temperature")

lin_ax.legend()

start, = np.where(datetimes.is_year_start)
year = datetimes[start[1]:start[2]]

season_ax.set_title("Seasonal effect on temperature")

season_ax.fill_between(year,
                    df.seasonality[year] * conf_int[0].seasonality,
                    df.seasonality[year] * conf_int[1].seasonality,
                    color = sns.color_palette()[1], alpha = 0.5, label="Average fluctuation (95% CI)")

season_ax.plot(year, np.zeros(year.shape), 
       linestyle="--", color = 'k',
       label = "Null Hypothesis")

season_ax.plot(year,df.seasonality[year] * parameters.seasonality,
            color = sns.color_palette()[1])

season_ax.set_xticks(year[year.is_month_start])
season_ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
                          rotation=45)
season_ax.set_ylabel("Mean seasonal temp fluctuation (Degree C)")
season_ax.legend()

ax.set_ylabel("Maximum temperature (Degree C)")
ax.set_xlabel("Year")
ax.set_title("Tasmanian Temperatures with prediction intervals")


fig.savefig("figures/model_output")