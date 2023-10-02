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













# ----------------------------------------------------


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
bike_model <- linear_reg() %>% # Type of model
  set_engine("lm")# Engine = What R function to use--linear model here

bike_workflow <- workflow() %>% 
  add_recipe(bike_recipe) %>%
  add_model(bike_model) %>%
  fit(data = log_bike_train) # Fit the workflow

# Look at fitted LM model
extract_fit_engine(bike_workflow) %>%
  tidy()
extract_fit_engine(bike_workflow) %>%
  summary

bike_predictions <- bind_cols(bike_test$datetime, 
                              predict(bike_workflow, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = exp(count)) %>% # Back-transform the log to original scale
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
  fit(data = log_bike_train) # Fit the workflow

pois_bike_predictions <- bind_cols(bike_test$datetime, 
                              predict(pois_bike_workflow, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = exp(count)) %>% # Back-transform the log to original scale
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=pois_bike_predictions, file="bike_pois_pred.csv", delim = ",")

# Penalized Linear Regression ------------------------
pen_lin_bike_model <- linear_reg(penalty = .01, mixture = .7) %>% # Set up model and tuning
  set_engine("glmnet") # Function to fit R

pen_lin_bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(pen_lin_bike_model) %>%
  fit(data = log_bike_train)

pen_lin_bike_predictions <- bind_cols(bike_test$datetime, 
                              predict(pen_lin_bike_workflow, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = exp(count)) %>% # Back-transform the log to original scale
  mutate(count = ifelse(count < 0, 0, count)) %>% # Make negative predictions into zeroes
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=pen_lin_bike_predictions, file="bike_pen_lin_pred.csv", delim = ",")

# Cross Validation ---------------------------------

# Penalized Linear Regression ------------------------
pen_lin_bike_model <- linear_reg(penalty = tune(), mixture = tune()) %>% # Set up model and tuning
  set_engine("glmnet") # Function to fit R

# Set Workflow
pen_lin_bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(pen_lin_bike_model)

# Grid of values to tune over
tuning_grid <- grid_regular(penalty(), mixture(), levels = 15) # levels = L means L^2 total tuning possibilities

# Split data for CV
folds <- vfold_cv(bike_train, v = 15, repeats = 1)

# Run the CV
CV_results <- pen_lin_bike_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq)) # or leave metrics NULL

# Plot Results (example)
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

# Find Best Tuning Parameters
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

#------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------
#------------------------------------------------------

# Separate Because the Recipe Changes

# In-Class Competition, 9/22/23 --------------------

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

bike_workflow <- workflow() %>% 
  add_recipe(bike_recipe) %>%
  add_model(bike_model) %>%
  fit(data = sqrt_bike_train) # Fit the workflow

# Look at fitted LM model
extract_fit_engine(bike_workflow) %>%
  tidy()
extract_fit_engine(bike_workflow) %>%
  summary

bike_predictions <- bind_cols(bike_test$datetime, 
                              predict(bike_workflow, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = (count ^ 2)) %>% # Back-transform the log to original scale
  mutate(count = ifelse(count < 0, 0, count)) %>% # Make negative predictions into zeroes
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
vroom_write(x=bike_predictions, file="class_competition.csv", delim = ",")



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
  set_mode("regression")

# Workflow w model and recipe -------------------
reg_tree_workflow <- workflow() %>% # Set Workflow
  add_recipe(bike_recipe) %>%
  add_model(reg_tree_model)

tuning_grid <- grid_regular(tree_depth(), # Grid of values to tune over
                            cost_complexity(), 
                            min_n(),
                            levels = 5) # levels = L means L^2 total tuning possibilities

folds <- vfold_cv(bike_train, # Split data for CV
                  v = 5, # 5 folds
                  repeats = 1)

CV_results <- reg_tree_workflow %>% # Run the CV
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq)) # or leave metrics NULL

# Find Best Tuning Parameters
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







# Random Forest --------------------------------

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
                              trees = 500) %>% # Type of Model
  set_engine("ranger") %>% # What R function to use
  set_mode("regression")


# Workflow w model and recipe -------------------
rand_for_workflow <- workflow() %>% # Set Workflow
  add_recipe(bike_recipe) %>%
  add_model(rand_for_model)

tuning_grid <- grid_regular(mtry(range = c(1, 9)), # Grid of values to tune over
                            min_n(),
                            levels = 2) # levels = L means L^2 total tuning possibilities

folds <- vfold_cv(bike_train, # Split data for CV
                  v = 2, # 5 folds
                  repeats = 1)

CV_results <- rand_for_workflow %>% # Run the CV
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq)) # or leave metrics NULL

# Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("rmse")

# Finalize the workflow and fit it
final_wf <-
  rand_for_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = log_bike_train)

# Predict
rand_for_bike_predictions <- bind_cols(bike_test$datetime, 
                              predict(final_wf, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = exp(count)) %>% # Back-transform the log to original scale
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=rand_for_bike_predictions, file="rand_for_bike_predictions.csv", delim = ",")







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

baked_bike_train %>% 
  slice(1:10) # Print First 10 Rows

# Cross Validation ------------------------
folds <- vfold_cv(bike_train, 
                  v = 5, 
                  repeats = 1) # Split data for CV

untuned_model <- control_stack_grid() # Control grid for tuning over a grid
tuned_model <- control_stack_resamples() # Control grid for models we aren't tuning

# Penalized Linear Regression Model -----------------------
pen_lin_reg <- linear_reg(penalty = tune(),
                          mixture = tune()) %>% # Set model and tuning
  set_engine("glmnet") # Function to fit R
  
pen_lin_reg_wf <- workflow() %>% # Set workflow
  add_recipe(bike_recipe) %>%
  add_model(pen_lin_reg)

pen_lin_reg_tg <- grid_regular(penalty(), # grid of values to tune over
                               mixture(),
                               levels = 5) # 25 tuning possibilities

pen_lin_reg_fit <- pen_lin_reg_wf %>% # Tune the model
  tune_grid(resamples = folds,
            grid = pen_lin_reg_tg,
            metrics = metric_set(rmse, mae),
            control = untuned_model)

# Regression Tree -------------------------
reg_tree <- decision_tree(tree_depth = tune(), #tune() means the computer will figure out the values later
                                cost_complexity = tune(),
                                min_n = tune()) %>% # We just set up the type of model
  set_engine("rpart") %>% # Engine = what R function to use
  set_mode("regression")

reg_tree_wf <- workflow() %>% # Set Workflow
  add_recipe(bike_recipe) %>%
  add_model(reg_tree)

reg_tree_tg <- grid_regular(tree_depth(), # Grid of values to tune over
                            cost_complexity(), 
                            min_n(),
                            levels = 5) # levels = L means L^2 total tuning possibilities

reg_tree_fit <- reg_tree_wf  %>% # Run the CV
  tune_grid(resamples = folds,
            grid = reg_tree_tg,
            metrics = metric_set(rmse, mae, rsq), # or leave metrics NULL
            control = untuned_model)

# Random Forest -----------------------------
rand_for <- rand_forest(mtry = tune(),
                              min_n = tune(),
                              trees = 500) %>% # Type of Model
  set_engine("ranger") %>% # What R function to use
  set_mode("regression")

rand_for_wf <- workflow() %>% # Set Workflow
  add_recipe(bike_recipe) %>%
  add_model(rand_for)

rand_for_tg <- grid_regular(mtry(range = c(1, 9)), # Grid of values to tune over
                            min_n(),
                            levels = 2) # levels = L means L^2 total tuning possibilities

rand_for_fit <- rand_for_wf %>% # Run the CV
  tune_grid(resamples = folds,
            grid = rand_for_tg,
            metrics = metric_set(rmse, mae, rsq),
            control = untuned_model) # or leave metrics NULL

# Stacked Model -------------------------------------
bike_stack <- stacks() %>% # Specify the models to include
  add_candidates(pen_lin_reg_fit) %>%
  add_candidates(reg_tree_fit) %>%
  add_candidates(rand_for_fit)

stacked_model <- bike_stack %>% # Fit the stacked model
  blend_predictions() %>% # LASSO penalized regression meta-learner
  fit_members() # Fit the members to the dataset

stacked_predictions <- bind_cols(bike_test$datetime,# Make predictions using stacked_model
                              predict(stacked_model, new_data = bike_test)) %>% # Bind predictions to corresponding datetime
  rename("datetime" = "...1", "count" = ".pred") %>% # Rename columns
  mutate(count = exp(count)) %>% # Back-transform the log to original scale
  mutate(datetime = as.character(format(datetime))) # Make datetime a character for vroom; otherwise there will be issues

# Comment this out because it writes our predictions to an Excel sheet and I don't want that to happen every time I run the script
# vroom_write(x=stacked_predictions, file="stacked_predictions.csv", delim = ",")
#-------------------------------------------------------------------------------------------