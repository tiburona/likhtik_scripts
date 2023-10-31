library(lme4)
library(nlme)
library(dplyr)
library(glmmTMB)
library(lmerTest)
library(sjPlot)
library(ggplot2)
library(emmeans)

### .3 second periods analyses ###

deviation_csv = paste('/Users/katie/likhtik/data/lfp/percent_freezing', 'theta_power_deviations.csv', sep='/')
deviation_data <- read.csv(deviation_csv, comment.char="#") 

data_with_rate <- deviation_data %>%
  filter(time_bin < 30) %>%
  group_by(period, group, period_type, neuron_type, unit, animal, trial) %>%
  summarise(
    hpc_theta_1_power = mean(hpc_theta_1_power, na.rm = TRUE),
    bla_theta_1_power = mean(bla_theta_1_power, na.rm = TRUE),
    pl_theta_1_power  = mean(pl_theta_1_power, na.rm = TRUE),
    rate = mean(rate, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )

#### BLA ###

bla_rate_power_model <- lmer(rate ~ group*period_type*neuron_type*bla_theta_1_power + (1|animal/unit), data = data_with_rate)
summary(bla_rate_power_model)

bla_predictions_data <- create_predictions_data(data_with_rate, bla_rate_power_model, 'bla_theta_1_power')
bla_plot <- graph_predictions(data=predictions, x='bla_theta_1_power', y='predicted', 
                       xlabel='BLA Theta 1 Power', ylabel='Predicted Firing Rate')

### PL ###

pl_rate_power_model <- lmer(rate ~ group*period_type*neuron_type*pl_theta_1_power + (1|animal/unit), data = data_with_rate)
summary(pl_rate_power_model)

pl_predictions_data <- create_predictions_data(data_with_rate, pl_rate_power_model, 'pl_theta_1_power')
pl_plot <- graph_predictions(data=predictions, x='pl_theta_1_power', y='predicted', 
                       xlabel='PL Theta 1 Power', ylabel='Predicted Firing Rate')



### HPC ###

hpc_rate_power_model <- lmer(rate ~ group*period_type*neuron_type*hpc_theta_1_power + (1|animal/unit), data = data_with_rate)
summary(hpc_rate_power_model)

hpc_predictions_data <- create_predictions_data(data_with_rate, hpc_rate_power_model, 'hpc_theta_1_power')
hpc_plot <- graph_predictions(data=predictions, x='hpc_theta_1_power', y='predicted', 
                             xlabel='HPC Theta 1 Power', ylabel='Predicted Firing Rate')



### Trying the .3 seconds of the last pip + .05, predicting firing rate during pip ###

previous_csv = paste('/Users/katie/likhtik/data/lfp/percent_freezing', 'previous_pip_power.csv', sep='/')
previous_data <- read.csv(previous_csv, comment.char="#")

previous_data_with_rate <- previous_data %>%
  filter(time_bin < 30) %>%
  group_by(period, group, period_type, neuron_type, unit, animal, trial) %>%
  summarise(
    mean_hpc_theta_1_power = mean(hpc_theta_1_power, na.rm = TRUE),
    mean_bla_theta_1_power = mean(bla_theta_1_power, na.rm = TRUE),
    mean_pl_theta_1_power  = mean(pl_theta_1_power, na.rm = TRUE),
    mean_rate = mean(rate, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )

bla_prev_rate_power_model <- lmer(mean_rate ~ group*period_type*neuron_type*mean_bla_theta_1_power + (1|animal/unit), data = previous_data_with_rate)
summary(bla_prev_rate_power_model)




hpc_prev_rate_power_model <- lmer(mean_rate ~ group*period_type*neuron_type*mean_hpc_theta_1_power + (1|animal/unit), data = previous_data_with_rate)
summary(hpc_prev_rate_power_model)

new_data <- expand.grid(
  mean_hpc_theta_1_power = c(mean(previous_data_with_rate$mean_hpc_theta_1_power) - sd(previous_data_with_rate$mean_hpc_theta_1_power), 
                             mean(previous_data_with_rate$mean_hpc_theta_1_power), 
                             mean(previous_data_with_rate$mean_hpc_theta_1_power) + sd(previous_data_with_rate$mean_hpc_theta_1_power)),
  group = unique(previous_data_with_rate$group),
  period_type = unique(previous_data_with_rate$period_type),
  neuron_type = unique(previous_data_with_rate$neuron_type)
)

new_data$predicted_rate <- predict(hpc_rate_power_model, newdata = new_data, re.form = NA)

ggplot(new_data, aes(x = mean_hpc_theta_1_power, y = predicted_rate, color = period_type)) +
  geom_line() +
  labs(x = "HPC Theta 1 Power", y = "Predicted Firing Rate") +
  theme_bw() +
  facet_grid(neuron_type ~ group, scales = "free")