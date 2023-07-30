library(glmmTMB)
library(lme4)
library(lmerTest)  # for lmer p-values
library(readr)
library(sjPlot)
library(ggplot2)
library(broom)
library(emmeans)

prepare_df <- function(csv, data_type, observation_type){
  df <- read_csv(csv)
  
  # Convert variables to factors
  factor_vars <- c('unit_num', 'animal', 'category', 'condition')
  df[factor_vars] <- lapply(df[factor_vars], factor)
  
  if (data_type == 'proportion' && observation_type == 'unit') {
    df$"proportion"[df$"proportion" == 0] <- df$"proportion"[df$"proportion" == 0] + 1e-6
  }
  
  
  return(df)
}

period_model <- function(data, period) {
  data = subset(data, data$period == period)
  model = lmer(rate~condition*category*power + (1|animal), data=period0_data_theta_1)
  return(summary(model))
}

plot_power_by_category_model <- function(data, model) {
  min_power <- min(data$power, na.rm = TRUE)
  max_power <- max(data$power, na.rm = TRUE)
  
  power_list <- list(power=seq(min_power,max_power,by=2.0))
  plot <- emmip(model, category ~ power, 
                at = list(category = c("IN", "PN"), 
                          power=seq(min_power, max_power, by=2.0)), 
                CIs = FALSE)
  return(plot + ylab("Predicted Normalized Firing Rate"))
}

analyze_brain_region_freq_band <- function(path_to_csv) {
  data = prepare_df(path_to_csv, 'psth', 'unit')
  omnibus_model = lmer(rate~condition*category*power + (1|animal/unit_num), data=data)
  period_fix_model = lmer(rate~condition*category*power*period + (1|animal), data=data)
  period_results <- lapply(0:4, function(x) period_model(data, x))
  power_by_category_model = lmer(rate ~ power*category + (1|animal/unit_num), data=data)
  plot = plot_power_by_category_model(data, power_by_category_model)
  
  return(list(
    data = data,
    omnibus_model = omnibus_model,
    period_fix_model = period_fix_model,
    period_results = period_results,
    power_by_category_model = power_by_category_model,
    power_by_category_plot = plot
  ))
}


### HPC ###
### Theta 1 ###

hpc_theta1 <- analyze_brain_region_freq_band('/Users/katie/likhtik/data/lfp/lfp_psth_hpc_theta_1_period_units.csv')
hpc_theta1$data
summary(hpc_theta1$omnibus_model)
summary(hpc_theta1$period_fix_model)
summary(hpc_theta1$power_by_category_model)
hpc_theta1$power_by_category_plot

### Theta 2 ###
  
hpc_theta2 <- analyze_brain_region_freq_band('/Users/katie/likhtik/data/lfp/lfp_psth_hpc_theta_2_period_units.csv')
hpc_theta2$data
summary(hpc_theta2$omnibus_model)
summary(hpc_theta2$period_fix_model)
summary(hpc_theta2$power_by_category_model)
hpc_theta2$power_by_category_plot

### PL ###
### Theta 1 ###

pl_theta1 <- analyze_brain_region_freq_band('/Users/katie/likhtik/data/lfp/lfp_psth_pl_theta_1_period_units.csv')
pl_theta1$data
summary(pl_theta1$omnibus_model)
summary(pl_theta1$period_fix_model)
summary(pl_theta1$power_by_category_model)
pl_theta1$power_by_category_plot




# 


