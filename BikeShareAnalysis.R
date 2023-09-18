# Bike Share Data Analysis
# 13 September 2023

# Load Libraries ---------------------------------------
library(tidyverse)
library(vroom)
library(tidymodels)
library(poissonreg)

# Load Data --------------------------------------------
bike_train <- vroom("train.csv")
bike_test <- vroom("test.csv")

# Data Cleaning ----------------------------------------
clean_train <- bike_train %>%
  select(-c(registered, casual)) # Remove casual and registered because they are direct components of count (count = casual + registered)
# Kept this out of the recipe because I only want to apply it to train.csv; test.csv doesn't even have registered and casual.
# Anything I want to do to both datasets should happen in the recipe.

# Feature Engineering ----------------------------------
bike_recipe <- recipe(count ~ ., data = clean_train) %>%
  step_mutate(weather = replace(weather, weather == 4, 3)) %>% # Replace the one instance of weather == 4 with weather == 3, which is similar
  step_num2factor(weather, levels = c("clear", "mist", "light_precip", "heavy_precip")) %>% # Make weather into a factor
  step_num2factor(season, levels = c("spring", "summer", "fall", "winter")) %>% # Make season into a factor
  step_num2factor(holiday, levels = c("non_holiday", "holiday"), transform = function(x) x + 1) %>% # Make holiday into a factor; numbers must be nonzero, so add 1 to each first
  step_num2factor(workingday, levels = c("non_workingday", "workingday"), transform = function(x) x + 1) %>% # Make workday into a factor; numbers must be nonzero, so add 1 to each first
  step_date(datetime, features = "dow") %>% # add a column for day of week
  step_time(datetime, features = "hour") %>% # add a column for hour
  step_rm(datetime) %>% # Remove datetime bc we have broken it into two more useful features--dow and hour
  step_rm(atemp) # Remove atemp bc it is multicollinear w temp
prepped_recipe <- prep(bike_recipe)
structured_train <- bake(prepped_recipe, new_data = clean_train)

# Print First 10 Rows ----------------------------------
structured_train %>%
  slice(1:10)

# Linear Regression ------------------------------------
bike_model <- linear_reg() %>% # Type of model
  set_engine("lm")# Engine = What R function to use--linear model here

bike_workflow <- workflow() %>% 
  add_recipe(bike_recipe) %>%
  add_model(bike_model) %>%
  fit(data = bike_train) # Fit the workflow

# Look at fitted LM model
extract_fit_engine(bike_workflow) %>%
  tidy()
extract_fit_engine(bike_workflow) %>%
  summary

bike_predictions <- bind_cols(bike_test$datetime, 
                              predict(bike_workflow, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = ifelse(count < 0, 0, count)) %>% # Make negative predictions into zeroes
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=bike_predictions, file="bike_lm_pred.csv", delim = ",")
 
# Poisson Regression ---------------------------------
pois_bike_model <- poisson_reg() %>% # Type of model
  set_engine("glm") # GLM = generalized linear model

pois_bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(pois_bike_model) %>%
  fit(data = bike_train) # Fit the workflow

pois_bike_predictions <- bind_cols(bike_test$datetime, 
                              predict(pois_bike_workflow, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=pois_bike_predictions, file="bike_pois_pred.csv", delim = ",")
