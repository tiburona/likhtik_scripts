library(lme4)
library(nlme)
library(dplyr)
library(glmmTMB)
library(lmerTest)
library(sjPlot)
library(ggplot2)
library(emmeans)
library(ggplot2)

run_rate_power_analysis <- function(data, region, frequency_band) {
  # Dynamically build the variable names based on region and frequency
  power_var <- paste(region, frequency_band, "power", sep = "_")
  
  # Filter out rows where the power variable is NA
  clean_data <- data[!is.na(data[[power_var]]), ]
  
  # Define control parameters for the linear mixed-effects model
  control = lmerControl(check.conv.grad=.makeCC("warning", tol=1e-4), optimizer="bobyqa")
  
  # Build the formula string dynamically
  formula <- as.formula(paste("rate ~ group*period_type*neuron_type*", power_var, 
                              "+ (1|animal/unit/period)", sep = ""))
  
  # Fit the model
  model <- lmer(formula, data = clean_data, control = control)
  
  # Display the summary of the model
  print(summary(model))
  
  # Calculate mean and standard deviation for the power variable
  mean_power <- mean(clean_data[[power_var]], na.rm = TRUE)
  sd_power <- sd(clean_data[[power_var]], na.rm = TRUE)
  
  pred_data <- expand.grid(
    group = levels(clean_data$group),
    period_type = levels(clean_data$period_type),
    neuron_type = levels(clean_data$neuron_type),
    power = c(mean_power - sd_power, mean_power, mean_power + sd_power)
  )
  
  # Properly name the power variable in the prediction data
  names(pred_data)[names(pred_data) == "power"] <- power_var
  
  pred_data$predicted_rate <- predict(model, newdata = pred_data, re.form = NA)
  
  
  # Plot the predictions
  p <- graph_predictions(data=pred_data, 
                         x=power_var, y='predicted_rate', 
                         xlabel=paste(toupper(region), frequency_band, "power", sep = " "), 
                         ylabel='Predicted Firing Rate')
  
  # Return both the model and the plot
  return(list(model = model, plot = p))
}




### .3 second periods analyses ###

csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/power', 'psth_power.csv', sep='/')
data <- read.csv(csv, comment.char="#") 

results <- data %>%
  filter(!is.na(bla_theta_1_power)) %>%
  group_by(animal, period, period_type, unit) %>%
  summarise(event_count = n_distinct(event), .groups = 'drop') %>%
  mutate(
    too_many_events = if_else(event_count > 30, "Yes", "No"),
    excluded_events = if_else(event_count < 30, "Yes", "No")
  )

excessive_events <- results %>%
  filter(too_many_events == "Yes")

excluded_events <- results %>%
  filter(excluded_events == "Yes")


factor_vars <- c('animal', 'group', 'period_type', 'neuron_type', 'unit')
data[factor_vars] <- lapply(data[factor_vars], factor)

data$neuron_type <- factor(data$neuron_type,
                           levels = c("IN", "PN"))
data <- data[!is.na(data$neuron_type), ]

data_with_rate <- data %>%
  filter(time_bin < 30) %>%
  group_by(period, group, period_type, neuron_type, unit, animal, event) %>%
  summarise(
    hpc_theta_1_power = mean(hpc_theta_1_power, na.rm = TRUE),
    bla_theta_1_power = mean(bla_theta_1_power, na.rm = TRUE),
    pl_theta_1_power  = mean(pl_theta_1_power, na.rm = TRUE),
    hpc_theta_2_power = mean(hpc_theta_2_power, na.rm = TRUE),
    bla_theta_2_power = mean(bla_theta_2_power, na.rm = TRUE),
    pl_theta_2_power  = mean(pl_theta_2_power, na.rm = TRUE),
    rate = mean(rate, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )

### BLA THETA 1 ###

results <- run_rate_power_analysis(data = data_with_rate, region = "bla", frequency_band = "theta_1")
print(results$plot)

qqnorm(resid(results$model))
qqline(resid(results$model), col = "red")


### BLA THETA 2 ###

results <- run_rate_power_analysis(data = data_with_rate, region = "bla", frequency_band = "theta_2")
print(results$plot)


### PL THETA 1 ###

results <- run_rate_power_analysis(data = data_with_rate, region = "pl", frequency_band = "theta_1")
print(results$plot)


### PL THETA 2 ###

results <- run_rate_power_analysis(data = data_with_rate, region = "pl", frequency_band = "theta_2")
print(results$plot)


### HPC THETA 1 ###

results <- run_rate_power_analysis(data = data_with_rate, region = "hpc", frequency_band = "theta_1")
print(results$plot)


### HPC THETA 2 ###

results <- run_rate_power_analysis(data = data_with_rate, region = "hpc", frequency_band = "theta_2")
print(results$plot)


