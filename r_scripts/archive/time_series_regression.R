library(lme4)
library(nlme)
library(dplyr)
library(glmmTMB)
library(lmerTest)
library(sjPlot)
library(ggplot2)



d = '/Users/katie/likhtik/data/lfp/power'
f = 'psth_power_pl_theta_1_power_pl_theta_2_power_bla_theta_1_power_bla_theta_2_power_hpc_theta_1_power_hpc_theta_2_percent_freezing_spike_power.csv'

csv_file = paste(d, f, sep='/')


data <- read.csv(csv_file, comment.char="#") 

factor_vars <- c('animal', 'group', 'neuron_type', 'period_type', 'unit')
data[factor_vars] <- lapply(data[factor_vars], factor)
data$transformed_trial = 30*data$period + data$trial

# Example models with different power variables as predictors
# power_vars <- c("power_pl_theta_1", "power_hpc_theta_1", "power_bla_theta_1")


# Normalizing function for min-max scaling
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Standardizing function for z-score normalization
standardize <- function(x) {
  return ((x - mean(x)) / sd(x))
}

# Apply the normalization and standardization functions to specified columns
data_transformed <- data %>%
  mutate(
    bla_theta_1_power = standardize(bla_theta_1_power),
    rate = normalize(rate)
  )

power_vars <- c("bla_theta_1_power")


# Initialize a data frame to store results
results <- data.frame()

# Loop through each power variable
for (power_var in power_vars) {
  
  # Initialize variables to store best model information
  best_aic <- Inf
  best_lag <- NA
  best_model <- NULL
  
  # Loop through potential lag values (e.g., 1 to 10)
  for (lag in 1:20) {
    
    # Create a lagged variable
    data <- data %>% mutate(lagged_power = lag(!!sym(power_var), lag))
    
    # Fit the model (example with fixed effects for 'group', 'neuron_type', and 'period_type', 
    # and random effects for 'animal', 'unit', 'period')
    zinb_mixed <- glmmTMB(rate ~ lagged_power + group + neuron_type + period_type + 
                            (1|animal) + (1|unit) + (1|transformed_trial),
                          ziformula = ~1 + (1|animal) + (1|unit) + (1|transformed_trial),
                          family = nbinom2, data = data)
    
    # Check if this model has a better (lower) AIC than the current best
    if (AIC(model) < best_aic) {
      best_aic <- AIC(model)
      best_lag <- lag
      best_model <- model
    }
  }
  
  # Store the best model results
  results <- rbind(results, data.frame(Predictor = power_var, BestLag = best_lag, BestAIC = best_aic))
}

# View the results
print(results)


data_lag_1 <- data_transformed %>% mutate(lagged_power = lag(data_transformed$bla_theta_1_power, 1))

zinb_mixed_lag_1 <- glmmTMB(rate ~ lagged_power + group + neuron_type + period_type + 
                        (1|animal) + (1|unit) + (1|transformed_trial),
                      ziformula = ~1 + (1|animal) + (1|unit) + (1|transformed_trial),
                      family = nbinom2, data = data_lag_1)


zinb_mixed_lag_0 <- glmmTMB(rate ~ bla_theta_1_power + group + neuron_type + period_type + 
                              (1|animal) + (1|unit) + (1|transformed_trial),
                            ziformula = ~1 + (1|animal) + (1|unit) + (1|transformed_trial),
                            family = nbinom2, data = data)

summary(zinb_mixed_lag_0)

zinb_mixed_lag_1_simplified <- glmmTMB(rate ~ lagged_power + group + neuron_type + period_type + 
                              (1|animal) + (1|unit) + (1|transformed_trial),
                            ziformula = ~1 + (1|animal) + (1|unit) + (1|transformed_trial),
                            family = nbinom2, data = data_lag_1)


data_lag_1 <- data_lag_1 %>%
  arrange(animal, unit, transformed_trial, time_bin) %>%
  mutate(global_time_point = row_number(),  
         lag_y = lag(rate, n = 65),  
         first_in_trial = ifelse(time_bin == 0, 1, 0))  


model <- glmmTMB(rate ~ lag_y + group * neuron_type * period_type + 
                   (1|animal/unit),
                 ziformula = ~1 + (1|animal/unit),
                 family = nbinom2, data = data_lag_1)


model <- glmer(rate ~ lag_y + group * neuron_type * period_type + 
                 (1|animal/unit), data = data_lag_1, family = poisson())

deviation_csv = paste('/Users/katie/likhtik/data/lfp/percent_freezing', 'theta_power_deviations.csv', sep='/')
deviation_data <- read.csv(moving_avg_csv, comment.char="#") 

# Assuming `your_data` is your data frame

# Get column names
col_names <- colnames(deviation_data)

# Iterate through each column name
for (col_name in col_names) {
  # Check if "deviation" is in the column name
  if (grepl("deviation", col_name)) {
    # Create a new column name for the indicator variable
    new_col_name <- paste0(col_name, "_indicator")
    
    # Create a new column based on the condition
    deviation_data[[new_col_name]] <- ifelse(deviation_data[[col_name]] >= 2, ">=2",
                                        ifelse(deviation_data[[col_name]] <= -2, "<=-2", "in between"))
  }
}

subset_data <- deviation_data %>%
  filter(bla_theta_1_power_deviation_indicator == ">=2" & time_bin < 30)


# Step 2: Identify columns with 'power' in their name but not 'indicator'
power_vars <- grep("power", names(subset_data), value = TRUE)
power_vars <- power_vars[!grepl("indicator", power_vars)]

# Step 3: Average over time bins and group by multiple variables
# Make sure to carry over variables that do not change within a trial
averaged_data <- subset_data %>%
  group_by(trial, group, animal, period, unit, period_type, neuron_type) %>%
  summarise(
    across(all_of(power_vars), mean, .names = "mean_{col}", na.rm = TRUE),
    rate_mean = mean(rate, na.rm = TRUE),
    .groups = "drop"
  )


model <- lmer(rate_mean ~ mean_bla_theta_1_power * period_type * neuron_type * group + (1|animal/unit), data=averaged_data)
summary(model)

new_data <- expand.grid(
  mean_bla_theta_1_power = c(mean(averaged_data$mean_bla_theta_1_power, na.rm=TRUE) - sd(averaged_data$mean_bla_theta_1_power, na.rm=TRUE), 
                      mean(averaged_data$mean_bla_theta_1_power, na.rm=TRUE), 
                      mean(averaged_data$mean_bla_theta_1_power, na.rm=TRUE) + sd(averaged_data$mean_bla_theta_1_power, na.rm=TRUE)),
  
  group = unique(averaged_data$group),
  period_type = unique(averaged_data$period_type),
  neuron_type = unique(averaged_data$neuron_type)
)

new_data$predicted_rate <- predict(model, newdata = new_data, re.form = NA)

ggplot(new_data, aes(x = bla_theta_1_mrl, y = predicted_rate, color = period_type)) +
  geom_line() +
  labs(x = "BLA Theta 1 Power", y = "Predicted Firing Rate") +
  theme_bw() +
  facet_grid(neuron_type ~ group, scales = "free")


simple_power_spike_csv =  paste('/Users/katie/likhtik/data/lfp/percent_freezing', 'theta_power_spike_by_period.csv', sep='/')
spike_power_data <- read.csv(simple_power_spike_csv, comment.char = "#")
factor_vars <- c('animal', 'group', 'neuron_type', 'period_type', 'unit')
spike_power_data[factor_vars] <- lapply(spike_power_data[factor_vars], factor)

model <- lmer(rate ~ pl_theta_2_power * period_type * neuron_type * group + (1|animal/unit), data=spike_power_data)
summary(model)


simple_power_spike_csv =  paste('/Users/katie/likhtik/data/lfp/percent_freezing', 'theta_power_spike_by_period.csv', sep='/')
spike_power_data <- read.csv(simple_power_spike_csv, comment.char = "#")
factor_vars <- c('animal', 'group', 'neuron_type', 'period_type', 'unit')
spike_power_data[factor_vars] <- lapply(spike_power_data[factor_vars], factor)

