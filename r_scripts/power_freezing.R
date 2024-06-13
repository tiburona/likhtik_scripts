library(lme4)
library(nlme)
library(dplyr)
library(glmmTMB)
library(lmerTest)
library(sjPlot)
library(ggplot2)
library(emmeans)


###BLA###

bla_freezing_power_model <- lmer(bla_theta_1_power ~ group*period_type*percent_freezing + (1|animal/period), 
                                 data = power_data)
summary(bla_freezing_power_model)

predictions <- create_predictions_data_no_nt(power_data, bla_freezing_power_model, 
                                       'percent_freezing')

p <- graph_predictions(data=predictions, x='percent_freezing', y='predicted', 
                       ylabel=paste('Predicted', 'BLA Theta 1 Power'), xlabel='Freezing', num_vars=3)


### PL ###

subset_data <- power_data[!is.na(power_data$percent_freezing) & !is.na(power_data$pl_theta_1_power), ]

pl_freezing_power_model <- lmer(pl_theta_1_power ~ group*period_type*percent_freezing + (1|animal/period), 
                                 data = subset_data)
summary(pl_freezing_power_model)

predictions <- create_predictions_data_no_nt(power_data, pl_freezing_power_model, 
                                             'percent_freezing')

p <- graph_predictions(data=predictions, x='percent_freezing', y='predicted', 
                       ylabel=paste('Predicted', 'PL Theta 1 Power'), xlabel='Percent Freezing', num_vars=3)




### HPC ###
subset_data <- power_data[!is.na(power_data$percent_freezing) & !is.na(power_data$hpc_theta_1_power), ]

hpc_freezing_power_model <- lmer(hpc_theta_1_power ~ group*period_type*percent_freezing + (1|animal/period), 
                                data = subset_data)
summary(hpc_freezing_power_model)

predictions <- create_predictions_data_no_nt(power_data, hpc_freezing_power_model, 
                                             'percent_freezing')

p <- graph_predictions(data=predictions, x='percent_freezing', y='predicted', 
                       ylabel=paste('Predicted', 'HPC Theta 1 Power'), xlabel='Percent Freezing', num_vars=3)



###BLA THETA 2###

subset_data <- power_data[!is.na(power_data$percent_freezing) & !is.na(power_data$bla_theta_2_power), ]

bla_freezing_power_model <- lmer(bla_theta_2_power ~ group*period_type*percent_freezing + (1|animal/period), 
                                 data = subset_data)
summary(bla_freezing_power_model)

predictions <- create_predictions_data_no_nt(power_data, bla_freezing_power_model, 
                                             'percent_freezing')

p <- graph_predictions(data=predictions, x='percent_freezing', y='predicted', 
                       ylabel=paste('Predicted', 'BLA Theta 2 Power'), xlabel='Freezing', num_vars=3)