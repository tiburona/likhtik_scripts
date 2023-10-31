library(lme4)
library(dplyr)
library(ggplot2)
library(emmeans)
library(rlang)
library(lmerTest)


csv_dir = '/Users/katie/likhtik/data/lfp/percent_freezing'
csv_name = 'psth_mrl_bla_delta_mrl_bla_t1Wu9w7standard_period_vals.csv'

csv_file = paste(csv_dir, csv_name, sep='/')
df <- read.csv(csv_file, comment.char="#") 
mrl_rate_df <- read.csv(csv_file, comment.char="#") 


read_metadata(csv_file)



mrl_rate_data <- mrl_rate_df %>%
  filter(time_bin < 30) %>%
  group_by(period, group, period_type, neuron_type, unit, animal, trial) %>%
  summarise(
    rate = mean(rate, na.rm = TRUE),
    hpc_theta_1_mrl = mean(hpc_theta_1_mrl, na.rm = TRUE),
    bla_theta_1_mrl = mean(bla_theta_1_mrl, na.rm = TRUE),
    pl_theta_1_mrl  = mean(pl_theta_1_mrl, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )

make_mrl_rate_model <- function(iv, data) {
  formula <- as.formula(paste("rate ~", iv, " * period_type * neuron_type * group + (1|animal/unit)"))
  model <- lmer(formula, data = data)
  print(summary(model))
  
  mean_iv <- mean(data[[iv]], na.rm = TRUE)
  sd_iv <- sd(data[[iv]], na.rm = TRUE)
  
  predictions_data <- create_predictions_data(data, model, iv)
  plot <- graph_predictions(predictions_data, iv, "predicted", iv, paste("Predicted", 'rate'))
  
  return(list(data = predictions_data, model = model, plot = plot))
}

variables <- c("hpc_theta_1_mrl", "bla_theta_1_mrl", "pl_theta_1_mrl")

results <- list()
for (i in seq_along(variables)) {
  iv <- variables[i]
  cat("Model and plot for IV:", iv, "\n")
  results[[iv]] <- make_mrl_rate_model(iv, mrl_rate_data)
}

summary(results[['hpc_theta_1_mrl']]$model)
results[['hpc_theta_1_mrl']]$plot

summary(results[['pl_theta_1_mrl']]$model)
results[['pl_theta_1_mrl']]$plot

summary(results[['bla_theta_1_mrl']]$model)
results[['bla_theta_1_mrl']]$plot


csv_dir = '/Users/katie/likhtik/data/lfp/percent_freezing'
csv_name = 'psth_mrl_bla_delta_mrl_bla_tqppvVVprevious_pip_lfp_05_firing_rate.csv'

csv_file = paste(csv_dir, csv_name, sep='/')
mrl_rate_df <- read.csv(csv_file, comment.char="#") 

read_metadata(csv_file)


mrl_rate_data <- mrl_rate_df %>%
  filter(time_bin < 30) %>%
  group_by(period, group, period_type, neuron_type, unit, animal, trial) %>%
  summarise(
    rate = mean(rate, na.rm = TRUE),
    hpc_theta_1_mrl = mean(hpc_theta_1_mrl, na.rm = TRUE),
    bla_theta_1_mrl = mean(bla_theta_1_mrl, na.rm = TRUE),
    pl_theta_1_mrl  = mean(pl_theta_1_mrl, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )

results <- list()
for (i in seq_along(variables)) {
  iv <- variables[i]
  cat("Model and plot for IV:", iv, "\n")
  results[[iv]] <- make_mrl_rate_model(iv, mrl_rate_data)
}

summary(results[['hpc_theta_1_mrl']]$model)
results[['hpc_theta_1_mrl']]$plot

summary(results[['pl_theta_1_mrl']]$model)
results[['pl_theta_1_mrl']]$plot

summary(results[['bla_theta_1_mrl']]$model)
results[['bla_theta_1_mrl']]$plot



