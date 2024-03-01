library(lme4)
library(nlme)
library(dplyr)
library(glmmTMB)
library(lmerTest)
library(sjPlot)
library(ggplot2)
library(emmeans)


power_csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/power', 'power_with_validated_pips_and_behavior.csv', sep='/')
power_data <- read.csv(power_csv, comment.char="#") 

power_data$animal = factor(power_data$animal)
power_data$period_type = factor(power_data$period_type)
power_data$group = factor(power_data$group)

power_data <- power_data %>%
  group_by(period, group, period_type, animal, event) %>%
  summarise(
    hpc_theta_1_power = mean(hpc_theta_1_power, na.rm = TRUE),
    bla_theta_1_power = mean(bla_theta_1_power, na.rm = TRUE),
    pl_theta_1_power  = mean(pl_theta_1_power, na.rm = TRUE),
    percent_freezing = mean(percent_freezing, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )


data_early_periods <- power_data %>%
  filter(period < 2)

###BLA###

bla_freezing_power_model <- lmer(bla_theta_1_power ~ group*period_type*percent_freezing + (1|animal/period), 
                                 data = power_data)
summary(bla_freezing_power_model)

predictions <- create_predictions_data(power_data, bla_freezing_power_model, 
                                       'percent_freezing', num_vars=3)

p <- graph_predictions(data=predictions, x='percent_freezing', y='predicted', 
                       ylabel='BLA Theta 1 Power', xlabel='Predicted Freezing', num_vars=3)


### PL ###

pl_freezing_power_model <- lmer(pl_theta_1_power ~ group*period_type*percent_freezing + (1|animal/period), 
                                 data = power_data)
summary(pl_freezing_power_model)

predictions <- create_predictions_data(power_data, pl_freezing_power_model, 
                                       'percent_freezing', num_vars=3)

p <- graph_predictions(data=predictions, x='percent_freezing', y='predicted', 
                       ylabel='PL Theta 1 Power', xlabel='Predicted Freezing', num_vars=3)


### HPC ###


hpc_freezing_power_model <- lmer(hpc_theta_1_power ~ group*period_type*percent_freezing + (1|animal/period), 
                                data = power_data)
summary(hpc_freezing_power_model)

predictions <- create_predictions_data(power_data, hpc_freezing_power_model, 
                                       'percent_freezing', num_vars=3)

p <- graph_predictions(data=predictions, x='percent_freezing', y='predicted', 
                       ylabel='HPC Theta 1 Power', xlabel='Predicted Freezing', num_vars=3)