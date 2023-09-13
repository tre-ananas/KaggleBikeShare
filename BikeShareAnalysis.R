# Bike Share Data Analysis
# 13 September 2023

# Load Libraries ---------------------------------------
library(tidyverse)
library(vroom)
library(tidymodels)

# Load Data --------------------------------------------
bike_train <- vroom("train.csv")
bike_test <- vroom("test.csv")

# Data Cleaning ----------------------------------------
clean_train <- bike_train %>%
  mutate(weather = replace(weather, weather == 4, 3)) # replace the one instance of weather == 4 with weather == 3, which is similar

# Feature Engineering ----------------------------------
bike_recipe <- recipe(count ~ ., data = clean_train) %>%
  step_num2factor(weather, levels = c("clear", "mist", "light_precip", "heavy_precip")) %>% # Make weather into a factor
  step_num2factor(season, levels = c("spring", "summer", "fall", "winter")) %>% # Make season into a factor
  step_num2factor(holiday, levels = c("non_holiday", "holiday"), transform = function(x) x + 1) %>% # Make holiday into a factor; numbers must be nonzero, so add 1 to each first
  step_num2factor(workingday, levels = c("non_workingday", "workingday"), transform = function(x) x + 1) %>% # Make workday into a factor; numbers must be nonzero, so add 1 to each first
  step_date(datetime, features = "dow") %>% # add a column for day of week
  step_time(datetime, features = "hour") %>% # add a column for hour
  step_rm(casual) %>% # Remove casual bc it is a direct component of count (count = casual + registered)
  step_rm(registered) %>% # Remove registered bc it is a direct component of count (count = casual + registered)
  step_rm(datetime) # Remove datetime bc we have broken it into two more useful features--dow and hour
prepped_recipe <- prep(bike_recipe)
structured_train <- bake(prepped_recipe, new_data = clean_train)

# Print First 10 Rows ----------------------------------
structured_train %>%
  slice(1:10)
