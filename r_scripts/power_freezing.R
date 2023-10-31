library(lme4)
library(nlme)
library(dplyr)
library(glmmTMB)
library(lmerTest)
library(sjPlot)
library(ggplot2)
library(emmeans)


deviation_csv = paste('/Users/katie/likhtik/data/lfp/percent_freezing', 'theta_power_deviations.csv', sep='/')
deviation_data <- read.csv(deviation_csv, comment.char="#") 

data_with_freezing <- deviation_data %>%
  filter(time_bin < 30) %>%
  group_by(period, group, period_type, neuron_type, animal) %>%
  summarise(
    hpc_theta_1_power = mean(hpc_theta_1_power, na.rm = TRUE),
    bla_theta_1_power = mean(bla_theta_1_power, na.rm = TRUE),
    pl_theta_1_power  = mean(pl_theta_1_power, na.rm = TRUE),
    rate = mean(rate, na.rm = TRUE),
    freezing = mean(percent_freezing, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )

###BLA###

bla_freezing_power_model <- lmer(freezing ~ group*period_type*bla_theta_1_power + (1|animal), 
                                 data = data_with_freezing)
summary(bla_freezing_power_model)

predictions <- create_predictions_data(data_with_freezing, bla_freezing_power_model, 
                                       'bla_theta_1_power', num_vars=3)

p <- graph_predictions(data=predictions, x='bla_theta_1_power', y='predicted', 
                       xlabel='BLA Theta 1 Power', ylabel='Predicted Freezing', num_vars=3)


### PL ###

pl_freezing_power_model <- lmer(freezing ~ group*period_type*pl_theta_1_power + (1|animal), 
                                 data = data_with_freezing)
summary(pl_freezing_power_model)

predictions <- create_predictions_data(data_with_freezing, pl_freezing_power_model, 
                                       'pl_theta_1_power', num_vars=3)

p <- graph_predictions(data=predictions, x='pl_theta_1_power', y='predicted', 
                       xlabel='PL Theta 1 Power', ylabel='Predicted Freezing', num_vars=3)


### HPC ###


hpc_freezing_power_model <- lmer(freezing ~ group*period_type*hpc_theta_1_power + (1|animal), 
                                data = data_with_freezing)
summary(hpc_freezing_power_model)

predictions <- create_predictions_data(data_with_freezing, hpc_freezing_power_model, 
                                       'hpc_theta_1_power', num_vars=3)

p <- graph_predictions(data=predictions, x='hpc_theta_1_power', y='predicted', 
                       xlabel='HPC Theta 1 Power', ylabel='Predicted Freezing', num_vars=3)