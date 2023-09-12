# Bike Share EDA
# 8 September 2023

# Load Libraries
library(tidyverse)
library(vroom)
library(DataExplorer)
library(patchwork)

#--------------------------------------------------------

# Load in Data from https://www.kaggle.com/c/bike-sharing-demand
bike <- vroom("train.csv")

#--------------------------------------------------------

# EDA

# Variable types
glimpse(bike) 
# datetime is date-time, but everything else is double
# Make relevant variables into factors
bike$season <- as_factor(bike$season)
bike$holiday <- as_factor(bike$holiday)
bike$workingday <- as_factor(bike$workingday)
bike$weather <- as_factor(bike$weather)
glimpse(bike) 
# all variables appear to be the right type now

# Visualization of glimpse()
plot_intro(bike)
# No missing observations or columns

# Numerical Overview of Variables
skimr::skim(bike) 
# 10,886 rows; 12 columns; no missing observations; 
# fairly evenly distributed seasons; far more non-holidays; 
# twice as many workdays; far and away mostly clear days 

plot_correlation(bike)
# Count (the response variable) is most corr. w/ season, humidity, and temp/atemp; little bit w/ weather_3 and weather_1
# Count's corr. w/ casual and registered doesn't count, really--these are count itself, just divided into two groups
# Good amount of multicollinearity going on:
# temp and atemp (feels like temp) are heavily correlated (obviously)
# casual, registered, and temp have moderate corr. w/ temp and atemp
# season and temp/atemp are obviously correlated
# weather and humidity are lightly to moderately correlated
# casual and registered use are corr. with workday status

# Bar Chart of Each Factor Frequency
plot_bar(bike)
# Seasons are pretty evenly distributed
# Overwhelmingly non-holiday--near zero variance
# Twice as many workdays as non-workdays
# More clear weather days (weather = 1) than the other 3 combined; almost no weather = 4 (heavy precipitation)

# Histograms of All Numerical Variables
plot_histogram(bike)
# casual, registered, and count are right skewed
# atemp and temp have a light gamma shape (but not a gamma range)
# humidity has an even lighter gamma shap and potentially some outliers on the low end
# windspeed has a huge cluster at 0, then a space, then the rest of the data

# Plot Missing Values
plot_missing(bike)
# Confirmed that there are no missing observations

# Mix of Plots - Keep this one commented out; it doesn't tell us many new things and seriously slows down R
# GGally::ggpairs(bike[, c("count", "temp", "atemp", "humidity", "windspeed")])
# count increases w temp and atemp; decreases w humidity; fairly ambiguous or light increase w/ windspeed

# Scatterplot of Total Bike Rentals Count by Temperature
count_by_temp <- ggplot(data = bike,
       mapping = aes(x = temp, y = count)) +
  geom_point(color = "skyblue", shape = 1) +
  geom_smooth(color = "navy", se = FALSE) +
  ggtitle("Number of Total Bike Rentals by Temperature") + 
  xlab("Temperature (C)") +
  ylab("Bike Rental Count") +
  theme(plot.title = element_text(hjust = .5))

# Scatterplot of Total Bike Rentals Count by Humidity
count_by_humidity <- ggplot(data = bike,
       mapping = aes(x = humidity, y = count)) +
  geom_point(shape = 1, color = "skyblue") +
  geom_smooth(se = FALSE, color = "navy") +
  ggtitle("Number of Total Bike Rentals by Relative Humidity") +
  xlab("Relative Humidity") +
  ylab("Bike Rental Count") +
  theme(plot.title = element_text(hjust = .5))

# Patchwork of Panel Plots
corr_plot <- plot_correlation(bike[, c("count", "temp", "humidity", "holiday","workingday")], title = "Correlation Plot for Select Variables")
plot_bar <- plot_bar(bike, title = "Frequency for Categorical/Factor Variables")

fourway_patch <- (corr_plot + plot_bar) / (count_by_temp + count_by_humidity)
fourway_patch
ggsave("BikeShareEDAFourwayPatch.png")
