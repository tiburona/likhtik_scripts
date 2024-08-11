library(lme4)
library(nlme)
library(dplyr)
library(glmmTMB)
library(lmerTest)
library(sjPlot)
library(ggplot2)
library(emmeans)
library(DHARMa)
library(boot)
library(purrr)
library(forcats)
library(caret)
library(brms)




# Function to plot aggregated actual data
plot_aggregated_data <- function(data, x, y, xlabel, ylabel, factors=c("neuron_type", "period_type", "group")) {
  # Create bins for the continuous IV based on quantiles
  
  data <- subset(data, !is.na(data[[x]]))
  
  data <- data %>%
    mutate(time_bin = ntile(get(x), 10))  # Creating 10 bins
  
  if ("period_type" %in% factors && "period_type" %in% names(data)) {
    data$period_type <- factor(data$period_type, levels = c("tone", "pretone"))
  }
  
 
  
  # Dynamically construct the grouping variables
  grouping_vars <- c("time_bin", "animal", "unit", "period", factors)
  
  # Compute averages at each level
  data <- data %>%
    group_by(across(all_of(grouping_vars))) %>%
    summarise(mean_y_period = median(get(y), na.rm = TRUE), .groups = 'drop') %>%
    group_by(across(setdiff(grouping_vars, "period"))) %>%
    summarise(mean_y_unit = median(mean_y_period, na.rm = TRUE), .groups = 'drop') %>%
    group_by(across(setdiff(grouping_vars, c("period", "unit")))) %>%
    summarise(mean_y_animal = median(mean_y_unit, na.rm = TRUE), .groups = 'drop') %>%
    group_by(across(setdiff(grouping_vars, c("period", "unit", "animal")))) %>%
    summarise(mean_y_group = median(mean_y_animal, na.rm = TRUE), .groups = 'drop')
  
  # Add a new interaction column for plotting
  data <- mutate(data, interaction_column = interaction(!!!syms(factors)))
  
  data <- mutate(data, color_for_period_type = gsub(".*\\b(pretone|tone)\\b.*", "\\1", interaction_column))
  
  
  
  # Setup plot aesthetics based on factors
  color_var <- if("period_type" %in% factors) "period_type" else "group"
  colors <- if(color_var == "period_type") {
    c("tone" = "green", "pretone" = "pink")
  } else {
    c("control" = "purple", "defeat" = "orange")
  }
  
  # Setup faceting based on factors
  facet_formula <- switch(length(factors),
                          '1' = paste0(". ~ ", factors),
                          '2' = paste0(factors[1], " ~ ", factors[2]),
                          '3' = "neuron_type ~ group",
                          '. ~ group')  # Default fall-back to free scale on group
  
  # Create the plot using the correct interaction column
  color_var <- if("period_type" %in% factors) "period_type" else "group"
  # Create the plot
  p <- ggplot(data, aes(x = as.factor(time_bin), y = mean_y_group, color = color_for_period_type, group = interaction_column)) +
    geom_line() +
    labs(x = xlabel, y = ylabel) +
    scale_color_manual(values = colors) +
    theme_bw() +
    facet_grid(facet_formula, scales = "free")
  
  return(p)
  
  
}






### .3 second periods analyses ###
  
csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/spike_counts', 'count_power_more_padding.csv', sep='/')
data <- read.csv(csv, comment.char="#") 

read_metadata(csv)


factor_vars <- c('animal', 'group', 'period_type', 'neuron_type', 'unit', 'period')
data[factor_vars] <- lapply(data[factor_vars], factor)

data$neuron_type <- factor(data$neuron_type,
                           levels = c("IN", "PN"))

data <- data[!is.na(data$neuron_type), ]


data$hpc_theta_1_power <- scale(data$hpc_theta_1_power)
data$bla_theta_1_power <- scale(data$bla_theta_1_power)
data$pl_theta_1_power  <- scale(data$pl_theta_1_power)
data$hpc_theta_2_power <- scale(data$hpc_theta_2_power)
data$bla_theta_2_power <- scale(data$bla_theta_2_power)
data$pl_theta_2_power  <- scale(data$pl_theta_2_power)



### PL THETA 1 ###

# Filter out rows where 'unit' is NA before summarizing
filtered_data <- data %>%
  filter(time < .3 & !is.na(unit))






# Proceed with grouping and summarization
continuous_data <- filtered_data %>%
  group_by(period, group, period_type, neuron_type, unit, animal, event, time) %>%
  summarise(
    hpc_theta_1_power = mean(hpc_theta_1_power, na.rm = TRUE),
    bla_theta_1_power = mean(bla_theta_1_power, na.rm = TRUE),
    pl_theta_1_power  = mean(pl_theta_1_power, na.rm = TRUE),
    hpc_theta_2_power = mean(hpc_theta_2_power, na.rm = TRUE),
    bla_theta_2_power = mean(bla_theta_2_power, na.rm = TRUE),
    pl_theta_2_power  = mean(pl_theta_2_power, na.rm = TRUE),
    count = sum(spike_counts, na.rm = TRUE),
    .groups = "drop"
  )

# Check for NA/Nan in the resulting data
post_na_check <- sum(is.na(continuous_data$unit))
post_nan_check <- sum(is.nan(continuous_data$unit))
print(paste("Post-summarisation NA in 'unit':", post_na_check))
print(paste("Post-summarisation NaN in 'unit':", post_nan_check))

# Filtering for complete cases in selected columns
na_rows_index <- !complete.cases(continuous_data[c("time", "group", "period_type", "neuron_type", "animal", "unit", "event")])
na_data <- continuous_data[na_rows_index, ]
head(na_data)  # Display the first few rows to inspect






continuous_data <- continuous_data %>%
  arrange(animal, unit, period, group, period_type, neuron_type, event, time) %>%
  group_by(animal, unit, period, group, period_type, neuron_type, event) %>%
  mutate(
    count_lag1 = lag(count, n = 1, default = NA),  
    count_lag2 = lag(count, n = 2, default = NA),
    count_lag3 = lag(count, n = 3, default = NA)
  ) %>%
  ungroup()



continuous_data <- continuous_data %>%
  filter(time >= 0) %>%
  ungroup()


continuous_data <- continuous_data %>%
  mutate(across(where(is.factor), droplevels))



pl_clean_data <- continuous_data[!is.na(continuous_data[['pl_theta_1_power']]), ]

na_rows_index <- !complete.cases(pl_clean_data[c("time_bin", "count_lag1", "group", "period_type", "neuron_type", "pl_theta_1_power", "animal", "unit", "event")])


na_data <- pl_clean_data[na_rows_index, ]
head(na_data)  # Display the first few rows to inspect



# plot_aggregated_data(clean_data, 'pl_theta_1_power', 'count', "PL Theta 1 Power", "Count", factors=c("neuron_type", "period_type", "group"))




install.packages("INLA", repos = c(getOption("repos"), INLA = "https://inla.r-inla-download.org/R/stable"))

pl_clean_data <- pl_clean_data %>%
  mutate(animal = as.factor(animal),
         unit = as.factor(unit),
         period = as.factor(period),
         event = as.factor(event),
         group = as.factor(group),
         period_type = as.factor(period_type),
         neuron_type = as.factor(neuron_type),
         pl_theta_1_power = as.numeric(pl_theta_1_power),
         count = as.numeric(count),
         time = as.numeric(time))  # Ensure time is numeric

# Create a nested time variable
pl_clean_data <- pl_clean_data %>%
  arrange(animal, unit, period, event, time) %>%
  group_by(animal, unit, period, event) %>%
  mutate(time_factor = as.numeric(row_number())) %>%  # Explicitly convert time_factor to numeric
  ungroup()

# Create a replicate identifier for the nested structure
pl_clean_data <- pl_clean_data %>%
  mutate(replicate_id = as.numeric(interaction(animal, unit, period, event, drop = TRUE)))

str(pl_clean_data)

pl_clean_data <- pl_clean_data %>%
  arrange(animal, unit, period, event, time) %>%
  group_by(animal, unit, period, event) %>%
  mutate(unique_time_num = as.numeric(row_number())) %>%
  ungroup()

# Define the correlation structure
corStruct <- corAR1(form = ~ unique_time_num | animal/unit/period/event)


# Fit the model
model <- lme(count ~ time + group * period_type * neuron_type * pl_theta_1_power,
             random = ~ 1 | animal/unit/period/event,
             correlation = corStruct,
             data = pl_clean_data,
             method = "REML")

# Summarize the model results
summary(model)

residuals <- resid(model)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

plot(residuals ~ fitted(model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

hist(residuals, main = "Histogram of Residuals")

formula <- bf(count ~ time + group * period_type * neuron_type * pl_theta_1_power + 
                ar(time = unique_time_num, gr = animal:unit:period:event) + (1| animal:unit) +
                (1 | animal:unit:period:event))

# Fit the model using brms
model <- brm(formula, data = pl_clean_data, family = zero_inflated_poisson(), chains = 4, cores = 4)

# Summarize the model results
summary(model)



model <- glmmTMB(count ~ time + count_lag1 + group * period_type * neuron_type * pl_theta_1_power + 
                   (1 | animal:unit) + (1 | animal:unit:period:event),
                 data = pl_clean_data, 
                 family = poisson(link = "log"),
                 ziformula = ~0,  # No zero-inflation
                 na.action = na.exclude,
                 control = glmmTMBControl(optimizer = optim, optArgs = list(method = "BFGS")))

simulationOutput <- simulateResiduals(fittedModel = model, plot = TRUE)

set.seed(123)  # for reproducibility
folds <- pl_clean_data %>%
  mutate(fold = sample(1:5, size = n(), replace = TRUE))

fit_glmmTMB <- function(train, test) {
  # Fit model
  model <- glmmTMB(count ~ time_bin + count_lag1 + group * period_type * neuron_type * pl_theta_1_power + 
                     (1 | animal:unit) + (1 | animal:unit:period:event),
                   data = train, 
                   family = poisson(link = "log"),
                   ziformula = ~0,  # Adjust according to your actual zero-inflation setup
                   control = glmmTMBControl(optimizer = optim, optArgs = list(method = "BFGS")))
  
  # Predict on test set
  pred <- predict(model, newdata = test, type = "response")
  
  # Return actual vs predicted
  data.frame(actual = test$count, predicted = pred)
}

# Perform cross-validation
results <- map_df(1:5, function(k) {
  train <- folds %>% filter(fold != k)
  test <- folds %>% filter(fold == k)
  
  fit_glmmTMB(train, test)
})

# Calculate performance metrics, e.g., RMSE
results %>% 
  mutate(residuals = actual - predicted,
         squared_residuals = residuals^2) %>%
  summarise(RMSE = sqrt(mean(squared_residuals)))


# Set up k-fold cross-validation
set.seed(123)  # for reproducibility
folds <- createFolds(pl_clean_data$count, k = 5, list = TRUE, returnTrain = TRUE)

# Function to perform model fitting and calculate RMSE for non-zero counts
perform_cv <- function(train_index, test_index) {
  train_data <- pl_clean_data[train_index, ]
  test_data <- pl_clean_data[test_index, ]
  
  # Fit the model
  model <- glmmTMB(count ~ time_bin + count_lag1 + group * period_type * neuron_type * pl_theta_1_power + 
                     (1 | animal:unit) + (1 | animal:unit:period:event),
                   data = train_data, 
                   family = poisson(link = "log"))
  
  # Predict on test set
  predictions <- predict(model, newdata = test_data, type = "response")
  
  # Obtain indices where actual counts are non-zero
  nonzero_indices <- which(test_data$count > 0)
  
  # Filter non-zero actual counts and corresponding predictions
  actual_nonzero <- test_data$count[nonzero_indices]
  predicted_nonzero <- predictions[nonzero_indices]
  
  # Calculate RMSE for non-zero counts
  rmse_nonzero <- sqrt(mean((actual_nonzero - predicted_nonzero)^2))
  
  return(rmse_nonzero)
}

# Apply the function to each fold
rmse_results <- sapply(folds, function(f) {
  train_index <- f
  test_index <- setdiff(seq_len(nrow(pl_clean_data)), train_index)
  perform_cv(train_index, test_index)
})

# Calculate average RMSE across folds
average_rmse_nonzero <- mean(rmse_results)
average_rmse_nonzero

model_without_time <- glmmTMB(count ~ count_lag1 + group * period_type * neuron_type * pl_theta_1_power + 
                                (1 | animal:unit) + (1 | animal:unit:period:event),
                              data = pl_clean_data, 
                              family = poisson(link = "log"),
                              ziformula = ~0,  # Adjust if you have zero-inflation
                              control = glmmTMBControl(optimizer = optim, optArgs = list(method = "BFGS")))

aic_with_time <- AIC(model)
aic_without_time <- AIC(model_without_time)


# Calculate mean and standard deviation for the power variable
mean_power <- mean(pl_clean_data[['pl_theta_1_power']], na.rm = TRUE)
sd_power <- sd(pl_clean_data[['pl_theta_1_power']], na.rm = TRUE)

# Create a combined identifier for the nested structure
pl_clean_data$animal_unit_period_event <- with(pl_clean_data, paste(animal, unit, period, event, sep = "_"))

# Generate combinations for freely varying factors
pred_data <- expand.grid(
  group = levels(pl_clean_data$group),
  period_type = levels(pl_clean_data$period_type),
  neuron_type = levels(pl_clean_data$neuron_type),
  power = c(mean_power - sd_power, mean_power, mean_power + sd_power)
)

# Merge with unique combinations of nested structure
unique_combinations <- unique(pl_clean_data[, c("animal_unit_period_event")])
pred_data <- merge(pred_data, unique_combinations, by = NULL) # Adjust based on how you want to include these nested identifiers

# Extract variables from the combined identifier
pred_data <- transform(pred_data, animal = sapply(strsplit(as.character(animal_unit_period_event), "_"), `[`, 1),
                       unit = as.integer(sapply(strsplit(as.character(animal_unit_period_event), "_"), `[`, 2)),
                       period = as.integer(sapply(strsplit(as.character(animal_unit_period_event), "_"), `[`, 3)),
                       event = as.integer(sapply(strsplit(as.character(animal_unit_period_event), "_"), `[`, 4)))

pred_data$time_bin <- mean(pl_clean_data$time_bin)
pred_data$count_lag1 <- mean(pl_clean_data$count_lag1)
pred_data$pl_theta_1_power <- pred_data$power

# Proceed with predictions
pred_data$predicted_counts <- predict(model, newdata = pred_data, re.form = NA, type = 'response')





# Plot the predictions
p <- graph_predictions(
  data=pred_data, x='pl_theta_1_power', y='predicted_counts', 
  xlabel= "pl theta 1 power",
  ylabel= labs(y = "Predicted Count of Spikes per Event (0-.3s)")) 





  # Ensure pl_theta_1_power is numeric and create bins for it
  grouped_data <- pl_clean_data %>%
    mutate(pl_theta_1_power = as.numeric(pl_theta_1_power),
           power_bin = ntile(pl_theta_1_power, 10))
  
  
  # Average over events
  grouped_data <- grouped_data %>%
    group_by(neuron_type, animal, unit, period, power_bin, period_type, group) %>%
    summarise(avg_count = mean(count, na.rm = TRUE), .groups = 'drop')
  
  # Average over periods
  grouped_data <- grouped_data %>%
    group_by(neuron_type, animal, unit, power_bin, period_type, group) %>%
    summarise(avg_count = mean(avg_count, na.rm = TRUE), .groups = 'drop')
  
  # Average over units
  grouped_data <- grouped_data %>%
    group_by(neuron_type, animal, power_bin, period_type, group) %>%
    summarise(avg_count = mean(avg_count, na.rm = TRUE), .groups = 'drop')
  
  # Average over animals
  grouped_data <- grouped_data %>%
    group_by(neuron_type, power_bin, period_type, group) %>%
    summarise(avg_count = mean(avg_count, na.rm = TRUE), .groups = 'drop')
  
  p <- ggplot(grouped_data, aes(x = as.factor(power_bin), y = avg_count, color = period_type)) +
    geom_point(aes(group = interaction(neuron_type, group, period_type))) +  # Ensure the same group interaction is used here
    geom_line(aes(group = interaction(neuron_type, group, period_type))) +  # This will connect points according to the group
    scale_color_manual(values = c("tone" = "green", "pretone" = "pink")) +
    facet_grid(neuron_type ~ group) +
    labs(x = "Binned pl_theta_1_power", y = "Average Count") +
    theme_bw()
  



plot_aggregated_data(pl_clean_data) 


# Example usage with your dataset
# plot_aggregated_data(your_dataset)




resid <- residuals(model, type = "response")
plot(resid, main = "Residuals Plot")

fitted_values <- fitted(model)
plot(fitted_values, resid, xlab = "Fitted values", ylab = "Residuals", main = "Residuals vs. Fitted Values")
abline(h = 0, col = "red")

observed_deviance <- sum(residuals(model, type = "pearson")^2)
df_residual <- df.residual(model)
overdispersion_factor = observed_deviance / df_residual
print(paste("Overdispersion factor:", overdispersion_factor))


# Simulate residuals using DHARMa
simulationOutput <- simulateResiduals(fittedModel = model)

plot(simulationOutput, quantreg = TRUE) 

# Alternatively, manually creating a Q-Q plot
# Predicted counts
fitted_counts <- fitted(model)

# Sort both observed counts and fitted counts
observed_sorted <- sort(model$data$count)
theoretical_quantiles <- qpois(ppoints(length(fitted_counts)), lambda = fitted_counts)

# Q-Q plot
plot(theoretical_quantiles, observed_sorted, main = "Q-Q Plot for Poisson Distribution",
     xlab = "Theoretical Quantiles", ylab = "Observed Quantiles")
abline(0, 1, col = "red")  # Line y = x for reference





### PL THETA 2 ###

clean_data <- data_with_counts[!is.na(data_with_counts[['pl_theta_2_power']]), ]
results <- run_count_power_analysis(data = clean_data, region = "pl", frequency_band = "theta_2")
print(results$plot)


### HPC THETA 1 ###

clean_data <- data_with_counts[!is.na(data_with_counts[['hpc_theta_1_power']]), ]
results <- run_count_power_analysis(data = clean_data, region = "hpc", frequency_band = "theta_1")
print(results$plot)


### HPC THETA 2 ###

clean_data <- data_with_counts[!is.na(data_with_counts[['hpc_theta_2_power']]), ]
results <- run_count_power_analysis(data = clean_data, region = "hpc", frequency_band = "theta_2")
print(results$plot)


