# Bike Share Data Analysis
# 13 September 2023

#-------------------------------------------------------------------------------------------
# SETUP - ALWAYS RUN THIS SECTION
#-------------------------------------------------------------------------------------------

# Load Libraries ---------------------------------------
library(tidyverse)
library(vroom)
library(tidymodels)
library(poissonreg)
library(rpart)
library(stacks)
library(dbarts)
library(xgboost)

# Load Data --------------------------------------------
bike_train <- vroom("train.csv")
bike_test <- vroom("test.csv")

# Data Cleaning ----------------------------------------
clean_train <- bike_train %>%
  select(-c(registered, casual)) # Remove casual and registered because they are direct components of count (count = casual + registered)
# Kept this out of the recipe because I only want to apply it to train.csv; test.csv doesn't even have registered and casual.
# Anything I want to do to both datasets should happen in the recipe.

# Transform to log(count)-------------------------------
# We only do this on the training set because the test set doesn't have count,
# hence we are working outside of the recipe because we only apply this to the training set
log_bike_train <- bike_train %>%
  mutate(count = log(count))

# Data Cleaning and Transformation
bike_train <- bike_train %>%
  select(-c(registered, casual)) %>% # Remove casual and registered because they are direct components of count (count = casual + registered)
  mutate(count = log(count)) # Transform count into log(count)

# Note on removing registered and casual from bike_train:
# Kept this out of the recipe because I only want to apply it to train.csv; test.csv doesn't even have registered and casual.
# Anything I want to do to both datasets should happen in the recipe.

# Note on transforming count to log(count):
# We only do this on the training set because the test set doesn't have count,
# hence we are working outside of the recipe because we only apply this to the training set

#-------------------------------------------------------------------------------------------

############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

#-------------------------------------------------------------------------------------------
# REGRESSIONS
#-------------------------------------------------------------------------------------------

# Feature Engineering ----------------------------------
bike_recipe <- recipe(count ~ ., data = clean_train) %>%
  step_mutate(weather = replace(weather, weather == 4, 3)) %>% # Replace the one instance of weather == 4 with weather == 3, which is similar
  # step_num2factor(weather, levels = c("clear", "mist", "light_precip", "heavy_precip")) %>% # Make weather into a factor
  # step_num2factor(season, levels = c("spring", "summer", "fall", "winter")) %>% # Make season into a factor
  # step_num2factor(holiday, levels = c("non_holiday", "holiday"), transform = function(x) x + 1) %>% # Make holiday into a factor; numbers must be nonzero, so add 1 to each first
  # step_num2factor(workingday, levels = c("non_workingday", "workingday"), transform = function(x) x + 1) %>% # Make workday into a factor; numbers must be nonzero, so add 1 to each first
  step_date(datetime, features = "dow") %>% # add a column for day of week
  step_time(datetime, features = "hour") %>% # add a column for hour
  step_rm(datetime) %>% # Remove datetime bc we have broken it into two more useful features--dow and hour
  step_rm(atemp) %>% # Remove atemp bc it is multicollinear w temp
  step_dummy(all_nominal_predictors()) %>% # Make nominal predictors into dummy variables
  step_normalize(all_numeric_predictors()) # Normalize numeric predictors to mean = 0, SD = 1 
prepped_recipe <- prep(bike_recipe)
structured_train <- bake(prepped_recipe, new_data = clean_train)

# Print First 10 Rows ----------------------------------
structured_train %>%
  slice(1:10)



# Linear Regression ------------------------------------
# Define model
bike_model <- linear_reg() %>% # Type of model
  set_engine("lm")# Engine = What R function to use--linear model here

# Fit workflow
bike_workflow <- workflow() %>% 
  add_recipe(bike_recipe) %>%
  add_model(bike_model) %>%
  fit(data = log_bike_train) # Fit the workflow

# Look at fitted LM model
extract_fit_engine(bike_workflow) %>%
  tidy()
extract_fit_engine(bike_workflow) %>%
  summary

# Predict outcomes
bike_predictions <- bind_cols(bike_test$datetime, 
                              predict(bike_workflow, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = exp(count)) %>% # Back-transform the log to original scale
  mutate(count = ifelse(count < 0, 0, count)) %>% # Make negative predictions into zeroes
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=bike_predictions, file="bike_lm_pred.csv", delim = ",")



# Poisson Regression ---------------------------------
# Define model
pois_bike_model <- poisson_reg() %>% # Type of model
  set_engine("glm") # GLM = generalized linear model

# Set up and fit workflow
pois_bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(pois_bike_model) %>%
  fit(data = log_bike_train) # Fit the workflow

# Make predictions
pois_bike_predictions <- bind_cols(bike_test$datetime, 
                              predict(pois_bike_workflow, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = exp(count)) %>% # Back-transform the log to original scale
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=pois_bike_predictions, file="bike_pois_pred.csv", delim = ",")



# Untuned Penalized Linear Regression ------------------------
# Define model and possible tuning parameters
pen_lin_bike_model <- linear_reg(penalty = .01, mixture = .7) %>% # Set up model and tuning parameters--not tuning this time
  set_engine("glmnet") # Function to fit R

# Set up and fit workflow
pen_lin_bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(pen_lin_bike_model) %>%
  fit(data = log_bike_train)

# Make predictions
pen_lin_bike_predictions <- bind_cols(bike_test$datetime, 
                              predict(pen_lin_bike_workflow, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = exp(count)) %>% # Back-transform the log to original scale
  mutate(count = ifelse(count < 0, 0, count)) %>% # Make negative predictions into zeroes
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=pen_lin_bike_predictions, file="bike_pen_lin_pred.csv", delim = ",")



# Penalized Linear Regression with Cross Validation ---------------------------------
# Define model and tuning parameters
pen_lin_bike_model <- linear_reg(penalty = tune(), mixture = tune()) %>% # Set up model and tuning--tune penalty and mixture
  set_engine("glmnet") # Function to fit R

# Set Workflow
pen_lin_bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(pen_lin_bike_model)

# Grid of values to tune over
tuning_grid <- grid_regular(penalty(), mixture(), levels = 15) # levels = L means L^2 total tuning possibilities

# Split data for CV
folds <- vfold_cv(bike_train, v = 15, repeats = 1) # 15 folds

# Run the CV
CV_results <- pen_lin_bike_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq)) # Collect rmse, mae, and rsq as metrics

# Plot Results for RMSE
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

# Find Best Tuning Parameters to minimize RMSE
bestTune <- CV_results %>%
  select_best("rmse")

# Finalize the workflow and fit it
final_wf <-
  pen_lin_bike_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = log_bike_train)

# Predict
cv_pen_lin_bike_predictions <- bind_cols(bike_test$datetime, 
                              predict(final_wf, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = exp(count)) %>% # Back-transform the log to original scale
  mutate(count = ifelse(count < 0, 0, count)) %>% # Make negative predictions into zeroes
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=cv_pen_lin_bike_predictions, file="cv_bike_pen_lin_pred.csv", delim = ",")

#-------------------------------------------------------------------------------------------

############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

#-------------------------------------------------------------------------------------------
# RANDOM FOREST 1
#-------------------------------------------------------------------------------------------

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

# Fit Model -------------------------------------
rand_for_model <- rand_forest(mtry = tune(),
                              min_n = tune(),
                              trees = 500) %>% # 500 trees, tune mtry and min_n
  set_engine("ranger") %>% # Use the ranger function
  set_mode("regression") # Regression bc outcome variable is quantitative


# Workflow w model and recipe -------------------
rand_for_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(rand_for_model)

# Set up tuning grid
tuning_grid <- grid_regular(mtry(range = c(1, 9)), # Grid of values to tune over
                            min_n(),
                            levels = 2) # levels = L means L^2 total tuning possibilities

# Cross validation with only 2 folds for speed
folds <- vfold_cv(bike_train, # Split data for CV
                  v = 2, # 2 folds
                  repeats = 1)

# Run the CV and store results
CV_results <- rand_for_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq)) # metrics: rmse, mae, rsq

# Find Best Tuning Parameters to Minimize rmse
bestTune <- CV_results %>%
  select_best("rmse")

# Finalize the workflow and fit it
final_wf <-
  rand_for_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = log_bike_train)

# Predict outcomes
rand_for_bike_predictions <- bind_cols(bike_test$datetime, 
                              predict(final_wf, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = exp(count)) %>% # Back-transform the log to original scale
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=rand_for_bike_predictions, file="rand_for_bike_predictions.csv", delim = ",")

#-------------------------------------------------------------------------------------------

############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

#-------------------------------------------------------------------------------------------
# MODEL STACKING
#-------------------------------------------------------------------------------------------

# Feature Engineering  ----------------------------------
bike_recipe <- recipe(count ~ ., data = bike_train) %>%
  step_mutate(weather = replace(weather, weather == 4, 3)) %>% # Replace the one instance of weather == 4 with weather == 3, which is similar
  step_num2factor(weather, levels = c("clear", "mist", "light_precip")) %>% # Make weather into a factor
  step_num2factor(season, levels = c("spring", "summer", "fall", "winter")) %>% # Make season into a factor
  step_num2factor(holiday, levels = c("non_holiday", "holiday"), transform = function(x) x + 1) %>% # Make holiday into a factor; numbers must be nonzero, so add 1 to each first
  step_num2factor(workingday, levels = c("non_workingday", "workingday"), transform = function(x) x + 1) %>% # Make workday into a factor; numbers must be nonzero, so add 1 to each first
  step_date(datetime, features = "dow") %>% # add a column for day of week
  step_time(datetime, features = "hour") %>% # add a column for hour
  step_rm(datetime) %>% # Remove datetime bc we have broken it into two more useful features--dow and hour
  step_rm(atemp) %>% # Remove atemp bc it is multicollinear w temp
  step_dummy(all_nominal_predictors()) %>% # Make nominal predictors into dummy variables
  step_zv() %>% # Remove columns with zero variance
  step_normalize(all_numeric_predictors()) # Normalize numeric predictors to mean = 0, SD = 1 
prepped_recipe <- prep(bike_recipe)
baked_bike_train <- bake(prepped_recipe, new_data = bike_train)

# Print First 10 Rows
baked_bike_train %>% 
  slice(1:10)



# Cross Validation ------------------------
folds <- vfold_cv(bike_train, 
                  v = 5, # 5 folds is a good balance of speed and effectiveness
                  repeats = 1) # Split data for CV

# Create control grids
untuned_model <- control_stack_grid() # Control grid for tuning over a grid
tuned_model <- control_stack_resamples() # Control grid for models we aren't tuning



# Penalized Linear Regression Model -----------------------
# Set up model
pen_lin_reg <- linear_reg(penalty = tune(),
                          mixture = tune()) %>% # Tune penalty and mixture
  set_engine("glmnet") # Function to fit R
  
# Set up workflow
pen_lin_reg_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(pen_lin_reg)

# Tuning grid for tuning penalty and mixture
pen_lin_reg_tg <- grid_regular(penalty(),
                               mixture(),
                               levels = 5) # 25 tuning possibilities

# Tune the model
pen_lin_reg_fit <- pen_lin_reg_wf %>%
  tune_grid(resamples = folds,
            grid = pen_lin_reg_tg,
            metrics = metric_set(rmse, mae), # Gather rmse and mae
            control = untuned_model) # Place in untuned_model grid



# Regression Tree -------------------------
# Set up model
reg_tree <- decision_tree(tree_depth = tune(), # tune() means the computer will figure out the values later
                                cost_complexity = tune(),
                                min_n = tune()) %>% # Tune tree depth, cost_complexity, and min_n
  set_engine("rpart") %>% # Engine = rpart, this is the function R uses
  set_mode("regression") # Regression bc variable we are predicting is quantitative

# Set up workflow
reg_tree_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(reg_tree)

# Tuning grid to tune cost_complexity, min_n, and tree_depth across 5 levels
reg_tree_tg <- grid_regular(tree_depth(),
                            cost_complexity(), 
                            min_n(),
                            levels = 5) # levels = L means L^2 total tuning possibilities

# Run CV
reg_tree_fit <- reg_tree_wf  %>%
  tune_grid(resamples = folds,
            grid = reg_tree_tg,
            metrics = metric_set(rmse, mae, rsq), # Metrics to collect: rmse, mae, rsq
            control = untuned_model)



# Random Forest -----------------------------
# Set up random forest model
rand_for <- rand_forest(mtry = tune(),
                              min_n = tune(),
                              trees = 500) %>% # 500 trees; tune mtry and min_n
  set_engine("ranger") %>% # R will use ranger as its function
  set_mode("regression") # Regression bc target variable is quantitative

# Set up workflow
rand_for_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(rand_for)

# Set up tuning grid
rand_for_tg <- grid_regular(mtry(range = c(1, 9)), # Grid of values to tune over
                            min_n(),
                            levels = 2) # Only 2 levels for speed

# Run CV
rand_for_fit <- rand_for_wf %>%
  tune_grid(resamples = folds,
            grid = rand_for_tg,
            metrics = metric_set(rmse, mae, rsq), # Metrics: rmse, mae, rsq
            control = untuned_model) # Assign to untuned_model control group



# Stacked Model -------------------------------------
# Set up the stacked model
bike_stack <- stacks() %>% # Specify the models to include
  add_candidates(pen_lin_reg_fit) %>%
  add_candidates(reg_tree_fit) %>%
  add_candidates(rand_for_fit)

# Fit the stacked model
stacked_model <- bike_stack %>% 
  blend_predictions() %>% # LASSO penalized regression meta-learner
  fit_members() # Fit the members to the dataset

# Make predictions with the stacked model
stacked_predictions <- bind_cols(bike_test$datetime,# Make predictions using stacked_model
                              predict(stacked_model, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = exp(count)) %>% # Back-transform the log to original scale
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=stacked_predictions, file="stacked_predictions.csv", delim = ",")
#-------------------------------------------------------------------------------------------

############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

#-------------------------------------------------------------------------------------------
# MORE RANDOM FORESTS
#-------------------------------------------------------------------------------------------

# Random Forest 2

# Feature Engineering ----------------------------------
bike_recipe <- recipe(count ~ ., data = bike_train) %>%
  step_mutate(weather = replace(weather, weather == 4, 3)) %>% # Replace the one instance of weather == 4 with weather == 3, which is similar
  step_num2factor(weather, levels = c("clear", "mist", "light_precip", "heavy_precip")) %>% # Make weather into a factor
  step_num2factor(season, levels = c("spring", "summer", "fall", "winter")) %>% # Make season into a factor
  step_num2factor(holiday, levels = c("non_holiday", "holiday"), transform = function(x) x + 1) %>% # Make holiday into a factor; numbers must be nonzero, so add 1 to each first
  step_num2factor(workingday, levels = c("non_workingday", "workingday"), transform = function(x) x + 1) %>% # Make workday into a factor; numbers must be nonzero, so add 1 to each first
  step_date(datetime, features = "dow") %>% # add a column for day of week
  step_time(datetime, features = "hour") %>% # add a column for hour
  step_num2factor(datetime_hour, levels = c("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                                            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                                            "nineteen", "twenty", "twenty-one", "twenty-two", "twenty-three"), transform = function(x) x + 1) %>%
  step_rm(datetime) %>% # Remove datetime bc we have broken it into two more useful features--dow and hour
  step_rm(atemp) # Remove atemp bc it is multicollinear w temp
prepped_recipe <- prep(bike_recipe)
bake(prepped_recipe, new_data = bike_train)

# Fit Model -------------------------------------
rand_for_model <- rand_forest(mtry = tune(),
                              min_n = tune(),
                              trees = 750) %>% # 750 trees, tune mtry and min_n
  set_engine("ranger") %>% # Use ranger function
  set_mode("regression")

# Workflow w model and recipe -------------------
rand_for_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(rand_for_model)

# Set up tuning grid
tuning_grid <- grid_regular(mtry(range = c(1, 9)), 
                            min_n(),
                            levels = 2) # 2 levels for speed

# 2 folds for CV--more speed, less effectiveness
folds <- vfold_cv(bike_train, # Split data for CV
                  v = 2,
                  repeats = 1)

# Run CV
CV_results <- rand_for_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq)) # Collect rmse, mae, and rsq metrics

# Find Best Tuning Parameters to minimize RMSE
bestTune <- CV_results %>%
  select_best("rmse")

# Finalize the workflow and fit it
final_wf <-
  rand_for_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = bike_train)

# Predict outcomes
rand_for_bike_predictions2 <- bind_cols(bike_test$datetime, 
                              predict(final_wf, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = exp(count)) %>% # Back-transform the log to original scale
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=rand_for_bike_predictions2, file="rand_for_bike_predictions2.csv", delim = ",")



# Random Forest 3

# Feature Engineering ----------------------------------
bike_recipe <- recipe(count ~ ., data = bike_train) %>%
  step_mutate(weather = replace(weather, weather == 4, 3)) %>% # Replace the one instance of weather == 4 with weather == 3, which is similar
  step_num2factor(weather, levels = c("clear", "mist", "light_precip", "heavy_precip")) %>% # Make weather into a factor
  step_num2factor(season, levels = c("spring", "summer", "fall", "winter")) %>% # Make season into a factor
  step_num2factor(holiday, levels = c("non_holiday", "holiday"), transform = function(x) x + 1) %>% # Make holiday into a factor; numbers must be nonzero, so add 1 to each first
  step_num2factor(workingday, levels = c("non_workingday", "workingday"), transform = function(x) x + 1) %>% # Make workday into a factor; numbers must be nonzero, so add 1 to each first
  step_date(datetime, features = "dow") %>% # add a column for day of week
  step_time(datetime, features = "hour") %>% # add a column for hour
  step_num2factor(datetime_hour, levels = c("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                                            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                                            "nineteen", "twenty", "twenty-one", "twenty-two", "twenty-three"), transform = function(x) x + 1) %>%
  step_rm(datetime) %>% # Remove datetime bc we have broken it into two more useful features--dow and hour
  step_rm(atemp) # Remove atemp bc it is multicollinear w temp
prepped_recipe <- prep(bike_recipe)
bake(prepped_recipe, new_data = bike_train)

# Fit Model -------------------------------------
rand_for_model <- rand_forest(mtry = tune(), # Tune mtry and min_n
                              min_n = tune(),
                              trees = 1000) %>% # 1000 trees
  set_engine("ranger") %>% # What R function to use
  set_mode("regression") # Regression for outcome variable


# Workflow w model and recipe -------------------
rand_for_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(rand_for_model)

# Set up tuning grid ----------------------------
tuning_grid <- grid_regular(mtry(range = c(1, 9)),
                            min_n(),
                            levels = 3) # 3 Levels

# Folds for cross validation
folds <- vfold_cv(bike_train,
                  v = 3, # 3 folds
                  repeats = 1)

# Run CV
CV_results <- rand_for_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse)) # Collect rmse metric

# Find Best Tuning Parameters to Minimize RMSE
bestTune <- CV_results %>%
  select_best("rmse")

# Finalize the workflow and fit it
final_wf <-
  rand_for_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = bike_train)

# Predict Outcomes
rand_for_bike_predictions3 <- bind_cols(bike_test$datetime, 
                              predict(final_wf, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = exp(count)) %>% # Back-transform the log to original scale
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=rand_for_bike_predictions3, file="rand_for_bike_predictions3.csv", delim = ",")



# Random Forest 4

# Feature Engineering ----------------------------------
bike_recipe <- recipe(count ~ ., data = bike_train) %>%
  step_mutate(weather = replace(weather, weather == 4, 3)) %>% # Replace the one instance of weather == 4 with weather == 3, which is similar
  step_num2factor(weather, levels = c("clear", "mist", "light_precip", "heavy_precip")) %>% # Make weather into a factor
  step_num2factor(season, levels = c("spring", "summer", "fall", "winter")) %>% # Make season into a factor
  step_num2factor(holiday, levels = c("non_holiday", "holiday"), transform = function(x) x + 1) %>% # Make holiday into a factor; numbers must be nonzero, so add 1 to each first
  step_num2factor(workingday, levels = c("non_workingday", "workingday"), transform = function(x) x + 1) %>% # Make workday into a factor; numbers must be nonzero, so add 1 to each first
  step_date(datetime, features = "dow") %>% # add a column for day of week
  step_time(datetime, features = "hour") %>% # add a column for hour
  step_num2factor(datetime_hour, levels = c("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                                            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                                            "nineteen", "twenty", "twenty-one", "twenty-two", "twenty-three"), transform = function(x) x + 1) %>%
  step_rm(datetime) %>% # Remove datetime bc we have broken it into two more useful features--dow and hour
  step_rm(atemp) %>% # Remove atemp bc it is multicollinear w temp
  step_rm(holiday) # Remove holiday because it has almost no entries and very low/no correlation w count
prepped_recipe <- prep(bike_recipe)
bake(prepped_recipe, new_data = bike_train)

# Fit Model -------------------------------------
rand_for_model <- rand_forest(mtry = tune(),
                              min_n = tune(),
                              trees = 1250) %>% # 1250 trees; tune mtry and min_n
  set_engine("ranger") %>% # R should use the ranger function
  set_mode("regression") # Regression bc outcome is quantitative


# Workflow w model and recipe -------------------
rand_for_workflow <- workflow() %>% 
  add_recipe(bike_recipe) %>%
  add_model(rand_for_model)

# Cross validation ---------------------
# Set up tuning grid
tuning_grid <- grid_regular(mtry(range = c(1, 8)), # Grid of values to tune over
                            min_n(),
                            levels = 5) # 5 levels

# 5 folds for CV
folds <- vfold_cv(bike_train, # Split data for CV
                  v = 5,
                  repeats = 1)

# Run CV
CV_results <- rand_for_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse)) # Collect RMSE

# Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("rmse") # Minimize RMSE

# Finalize the workflow and fit it
final_wf <-
  rand_for_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = bike_train)

# Predict outcomes
rand_for_bike_predictions4 <- bind_cols(bike_test$datetime, 
                              predict(final_wf, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = exp(count)) %>% # Back-transform the log to original scale
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=rand_for_bike_predictions4, file="rand_for_bike_predictions4.csv", delim = ",")



# Random Forest 5

# Feature Engineering ----------------------------------
bike_recipe <- recipe(count ~ ., data = bike_train) %>%
  step_mutate(weather = replace(weather, weather == 4, 3)) %>% # Replace the one instance of weather == 4 with weather == 3, which is similar
  step_num2factor(weather, levels = c("clear", "mist", "light_precip", "heavy_precip")) %>% # Make weather into a factor
  step_num2factor(season, levels = c("spring", "summer", "fall", "winter")) %>% # Make season into a factor
  step_num2factor(holiday, levels = c("non_holiday", "holiday"), transform = function(x) x + 1) %>% # Make holiday into a factor; numbers must be nonzero, so add 1 to each first
  step_num2factor(workingday, levels = c("non_workingday", "workingday"), transform = function(x) x + 1) %>% # Make workday into a factor; numbers must be nonzero, so add 1 to each first
  step_date(datetime, features = "dow") %>% # add a column for day of week
  step_time(datetime, features = "hour") %>% # add a column for hour
  step_num2factor(datetime_hour, levels = c("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                                            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                                            "nineteen", "twenty", "twenty-one", "twenty-two", "twenty-three"), transform = function(x) x + 1) %>%
  step_rm(datetime) %>% # Remove datetime bc we have broken it into two more useful features--dow and hour
  step_interact(~ temp:humidity) %>%
  step_corr(threshold = .9) %>%
  step_zv()
prepped_recipe <- prep(bike_recipe)
x <- bake(prepped_recipe, new_data = bike_train)

# Fit Model -------------------------------------
rand_for_model <- rand_forest(mtry = tune(),
                              min_n = tune(),
                              trees = 1000) %>% # 1000 trees; tune mtry and min_n
  set_engine("ranger") %>% # R will use the ranger function
  set_mode("regression") # regression bc the outcome variable is quantitative


# Workflow w model and recipe -------------------
rand_for_workflow <- workflow() %>% # Set Workflow
  add_recipe(bike_recipe) %>%
  add_model(rand_for_model)

# Cross validation ------------------
# Set up tuning grid
tuning_grid <- grid_regular(mtry(range = c(1, 9)), # Grid of values to tune over
                            min_n(),
                            levels = 10) # 10 levels

# Split data for CV
folds <- vfold_cv(bike_train,
                  v = 10, # 10 folds
                  repeats = 1)

# Run CV
CV_results <- rand_for_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse)) # collect rmse metric

# Find Best Tuning Parameters to Minimize RMSE
bestTune <- CV_results %>%
  select_best("rmse")

# Finalize the workflow and fit it
final_wf <-
  rand_for_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = bike_train)

# Predict outcomes
rand_for_bike_predictions5 <- bind_cols(bike_test$datetime, 
                              predict(final_wf, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = exp(count)) %>% # Back-transform the log to original scale
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=rand_for_bike_predictions5, file="rand_for_bike_predictions5.csv", delim = ",")

#-------------------------------------------------------------------------------------------

############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

#------------------------------------------------------------------------------------------------------------------
# Bayesian Additive Regression Trees (BART)
#------------------------------------------------------------------------------------------------------------------

# Feature Engineering ----------------------------------
bike_recipe <- recipe(count ~ ., data = bike_train) %>%
  step_mutate(weather = replace(weather, weather == 4, 3)) %>% # Replace the one instance of weather == 4 with weather == 3, which is similar
  step_num2factor(weather, levels = c("clear", "mist", "light_precip", "heavy_precip")) %>% # Make weather into a factor
  step_num2factor(season, levels = c("spring", "summer", "fall", "winter")) %>% # Make season into a factor
  step_num2factor(holiday, levels = c("non_holiday", "holiday"), transform = function(x) x + 1) %>% # Make holiday into a factor; numbers must be nonzero, so add 1 to each first
  step_num2factor(workingday, levels = c("non_workingday", "workingday"), transform = function(x) x + 1) %>% # Make workday into a factor; numbers must be nonzero, so add 1 to each first
  step_date(datetime, features = "year") %>% # add a column for year
  step_date(datetime, features = "month") %>% # add a column for month
  step_date(datetime, features = "dow") %>% # add a column for day of week
  step_time(datetime, features = "hour") %>% # add a column for hour
  step_num2factor(datetime_hour, levels = c("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                                            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                                            "nineteen", "twenty", "twenty-one", "twenty-two", "twenty-three"), transform = function(x) x + 1) %>% # Turn hour into a factor
  step_rm(datetime) %>% # Remove datetime bc we have broken it into two more useful features--dow and hour
  step_rm(atemp) # Remove atemp bc it is multicollinear w temp
prepped_recipe <- prep(bike_recipe)
x <- bake(prepped_recipe, new_data = bike_train)

# Fit Model -------------------------------------
bart_model <- bart(
  mode = "regression", # regresion bc outcome is quantitative
  engine = "dbarts", # R will use the dbarts function
  trees = 500, # 500 trees
  prior_terminal_node_coef = tune(), # We will tune these
  prior_terminal_node_expo = tune(),
  prior_outcome_range = tune()
)

# Set workflow w model and recipe -------------------
bart_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(bart_model)

# CV ------------------------------
# Set up tuning grid
tuning_grid <- grid_regular(prior_terminal_node_coef(range = c(0.01, 1.0)), # Grid of values to tune over
                            prior_terminal_node_expo(range = c(0.01, 4.0)),
                            prior_outcome_range(range = c(-3, 3)),
                            levels = 2)

# Split data into folds for CV
folds <- vfold_cv(bike_train,
                  v = 2, # 2 folds
                  repeats = 1)

# Run CV
CV_results <- bart_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq)) # Collect rmse, mae, and rsq

# Find Best Tuning Parameters to Minimize RMSE
bestTune <- CV_results %>%
  select_best("rmse")

# Finalize the workflow and fit it
final_wf <-
  bart_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = bike_train)

# Predict outcomes
bart_predictions <- bind_cols(bike_test$datetime, 
                              predict(final_wf, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = exp(count)) %>% # Back-transform the log to original scale
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=bart_predictions, file="bart_predictions.csv", delim = ",")

#-------------------------------------------------------------------------------------------

############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

#------------------------------------------------------------------------------------------------------------------
# Boosted Trees
#------------------------------------------------------------------------------------------------------------------

# Feature Engineering ----------------------------------
bike_recipe <- recipe(count ~ ., data = bike_train) %>%
  step_mutate(weather = replace(weather, weather == 4, 3)) %>% # Replace the one instance of weather == 4 with weather == 3, which is similar
  step_num2factor(weather, levels = c("clear", "mist", "light_precip", "heavy_precip")) %>% # Make weather into a factor
  step_num2factor(season, levels = c("spring", "summer", "fall", "winter")) %>% # Make season into a factor
  step_num2factor(holiday, levels = c("non_holiday", "holiday"), transform = function(x) x + 1) %>% # Make holiday into a factor; numbers must be nonzero, so add 1 to each first
  step_num2factor(workingday, levels = c("non_workingday", "workingday"), transform = function(x) x + 1) %>% # Make workday into a factor; numbers must be nonzero, so add 1 to each first
  step_date(datetime, features = "year") %>% # add a column for year
  step_date(datetime, features = "month") %>% # add a column for month
  step_date(datetime, features = "dow") %>% # add a column for day of week
  step_time(datetime, features = "hour") %>% # add a column for hour
  step_num2factor(datetime_hour, levels = c("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                                            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                                            "nineteen", "twenty", "twenty-one", "twenty-two", "twenty-three"), transform = function(x) x + 1) %>% # Turn hour into a factor
  step_rm(datetime) %>% # Remove datetime bc we have broken it into two more useful features--dow and hour
  step_rm(atemp) # Remove atemp bc it is multicollinear w temp
prepped_recipe <- prep(bike_recipe)
bake(prepped_recipe, new_data = bike_train)

# Fit Model -------------------------------------
boost_model <- boost_tree(mode = "regression", # Predict a quantitative variable using xgboost
                          engine = "xgboost")


# Set workflow w model and recipe -------------------
bart_wf <- workflow() %>% 
  add_recipe(bike_recipe) %>%
  add_model(bart_model)

# Cross Validation -------------------
# Set up tuning grid
tuning_grid <- grid_regular(prior_terminal_node_coef(range = c(0.01, 1.0)), # Grid of values to tune over
                            prior_terminal_node_expo(range = c(0.01, 4.0)),
                            prior_outcome_range(range = c(-3, 3)),
                            levels = 2) # 2 levels for speed

# Split data for CV
folds <- vfold_cv(bike_train,
                  v = 2, # 2 folds
                  repeats = 1)

# Run the CV
CV_results <- bart_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq)) # Collect rmse, mae, and rsq

# Find Best Tuning Parameters to Minimize RMSE
bestTune <- CV_results %>%
  select_best("rmse")

# Finalize the workflow and fit it
final_wf <-
  bart_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = bike_train)

# Predict Outcomes
bart_predictions <- bind_cols(bike_test$datetime, 
                              predict(final_wf, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = exp(count)) %>% # Back-transform the log to original scale
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=bart_predictions, file="bart_predictions.csv", delim = ",")

#-------------------------------------------------------------------------------------------

############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

#-------------------------------------------------------------------------------------------
# IN CLASS COMPETITION
#-------------------------------------------------------------------------------------------

# In-Class Competition, 9/22/23

# Transform to sqrt(count)-------------------------------
# We only do this on the training set because the test set doesn't have count,
# hence we are working outside of the recipe because we only apply this to the training set
sqrt_bike_train <- bike_train %>%
  mutate(count = sqrt(count))

# Feature Engineering ----------------------------------
bike_recipe <- recipe(count ~ ., data = clean_train) %>%
  step_mutate(weather = replace(weather, weather == 4, 3)) %>% # Replace the one instance of weather == 4 with weather == 3, which is similar
  step_mutate(poly_humidity = -(humidity^2)) %>%
  step_num2factor(weather, levels = c("clear", "mist", "light_precip", "heavy_precip")) %>% # Make weather into a factor
  step_num2factor(season, levels = c("spring", "summer", "fall", "winter")) %>% # Make season into a factor
  step_num2factor(holiday, levels = c("non_holiday", "holiday"), transform = function(x) x + 1) %>% # Make holiday into a factor; numbers must be nonzero, so add 1 to each first
  step_num2factor(workingday, levels = c("non_workingday", "workingday"), transform = function(x) x + 1) %>% # Make workday into a factor; numbers must be nonzero, so add 1 to each first
  step_date(datetime, features = "dow") %>% # add a column for day of week
  step_time(datetime, features = "hour") %>% # add a column for hour
  step_rm(datetime) %>% # Remove datetime bc we have broken it into two more useful features--dow and hour
  step_rm(atemp) %>% # Remove atemp bc it is multicollinear w temp
  step_dummy(all_nominal_predictors()) %>% # Make nominal predictors into dummy variables
  step_zv(all_predictors()) %>% # Remove columns with zero variance
  step_normalize(all_numeric_predictors()) %>% # Normalize numeric predictors to mean = 0, SD = 1 
  step_rm(datetime_dow_Fri) # Fix rank-deficiency, which I believe comes from multicollinearity
prepped_recipe <- prep(bike_recipe)
structured_train <- bake(prepped_recipe, new_data = clean_train)

# Print First 10 Rows ----------------------------------
structured_train %>%
  slice(1:10)

# Linear Regression ------------------------------------
bike_model <- linear_reg() %>% # Type of model
  set_engine("lm")# Engine = What R function to use--linear model here

# Set up and fit workflow
bike_workflow <- workflow() %>% 
  add_recipe(bike_recipe) %>%
  add_model(bike_model) %>%
  fit(data = sqrt_bike_train) # Fit the workflow

# Look at fitted LM model
extract_fit_engine(bike_workflow) %>%
  tidy()
extract_fit_engine(bike_workflow) %>%
  summary

# make Predictions
bike_predictions <- bind_cols(bike_test$datetime, 
                              predict(bike_workflow, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = (count ^ 2)) %>% # Back-transform the log to original scale
  mutate(count = ifelse(count < 0, 0, count)) %>% # Make negative predictions into zeroes
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=bike_predictions, file="class_competition.csv", delim = ",")



# Regression Tree --------------------------------

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

# Fit Model -------------------------------------
reg_tree_model <- decision_tree(tree_depth = tune(), # tune() means the computer will figure out the values later
                                cost_complexity = tune(),
                                min_n = tune()) %>% # We just set up the type of model
  set_engine("rpart") %>% # Engine = what R function to use
  set_mode("regression") # Regression because the variable we're predicting is quantitative

# Workflow w model and recipe -------------------
reg_tree_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(reg_tree_model)

# Set up tuning grid
tuning_grid <- grid_regular(tree_depth(), # Grid of values to tune over
                            cost_complexity(), 
                            min_n(),
                            levels = 5) # levels = L means L^2 total tuning possibilities

# 5 fold cross validation
folds <- vfold_cv(bike_train, # Split data for CV
                  v = 5, # 5 folds
                  repeats = 1)

# Run the CV for metrics rmse, mae, and rsq
CV_results <- reg_tree_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq))

# Find Best Tuning Parameters to minimize RMSE
bestTune <- CV_results %>%
  select_best("rmse")

# Finalize the workflow and fit it
final_wf <-
  reg_tree_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = log_bike_train)

# Predict
reg_tree_bike_predictions <- bind_cols(bike_test$datetime, 
                              predict(final_wf, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = exp(count)) %>% # Back-transform the log to original scale
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=reg_tree_bike_predictions, file="reg_tree_bike_predictions.csv", delim = ",")

############################################################################################
############################################################################################
# END OF CODE
############################################################################################
############################################################################################