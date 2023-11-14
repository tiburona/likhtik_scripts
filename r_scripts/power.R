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

read_metadata(deviation_csv)


power_data <- deviation_data %>%
  filter(time_bin < 30) %>%
  group_by(period, group, period_type, animal) %>%
  summarise(
    hpc_theta_1_power = mean(hpc_theta_1_power, na.rm = TRUE),
    bla_theta_1_power = mean(bla_theta_1_power, na.rm = TRUE),
    pl_theta_1_power  = mean(pl_theta_1_power, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )

data_early_periods <- power_data %>%
  filter(period < 2)

plot_values_over_periods <- function(data, y_label, y_var) {
  ggplot(data, aes_string(x = "period", y = y_var, color = "period_type", shape = "period_type")) +
    stat_summary(fun = mean, geom = "point", size = 3, aes(shape = period_type, color = period_type)) +
    stat_summary(fun = mean, geom = "line", aes(group = period_type)) +
    scale_shape_manual(values = c("pretone" = 15, "tone" = 16)) +
    scale_color_manual(values = c("pretone" = "gray", "tone" = "blue")) +
    labs(x = "Period", y = y_label) +
    facet_wrap(~ group, ncol = 1) +
    theme_minimal()
}



# BLA #
bla_plot <- plot_values_over_periods(power_data, 'Mean BLA Theta 1 Power', 'bla_theta_1_power')



# PL #

pl_plot <- plot_values_over_periods(power_data, 'Mean PL Theta 1 Power', 'pl_theta_1_power')


# HPC #

hpc_plot <- plot_values_over_periods(power_data, 'Mean HPC Theta 1 Power', 'hpc_theta_1_power')


### Models ####


bla_model <- lmer(bla_theta_1_power ~ group * period_type + (1|animal), data=power_data)
summary(bla_model)
bla_plot = emmip(bla_model, group ~ period_type, CIs = FALSE) + 
  labs(y = "Predicted BLA Power") +
  scale_color_manual(values = c("control" = "green", "stressed" = "orange"))

bla_early_period_model <- lmer(bla_theta_1_power ~ group * period_type + (1|animal), data=data_early_periods)
summary(bla_early_period_model)
bla_early_plot = emmip(bla_early_period_model, group ~ period_type, CIs = FALSE) + 
  labs(y = "Predicted PL Power") +
  scale_color_manual(values = c("control" = "green", "stressed" = "orange"))


pl_model <- lmer(pl_theta_1_power ~ group * period_type + (1|animal), data=power_data)
summary(pl_model)
pl_plot = emmip(pl_model, group ~ period_type, CIs = FALSE) + 
  labs(y = "Predicted PL Power") +
  scale_color_manual(values = c("control" = "green", "stressed" = "orange"))

pl_early_period_model <- lmer(mean_pl_theta_1_power ~ group * period_type + (1|animal), data=data_early_periods)
summary(pl_early_period_model)
pl_early_plot = emmip(pl_early_period_model, group ~ period_type, CIs = FALSE) + 
  labs(y = "Predicted PL Power (first two periods)") +
  scale_color_manual(values = c("control" = "green", "stressed" = "orange"))


hpc_model <- lmer(mean_hpc_theta_1_power ~ group * period_type + (1|animal), data=mean_data)
summary(hpc_model)
hpc_plot = emmip(hpc_model, group ~ period_type, CIs = FALSE) + 
  labs(y = "Predicted HPC Power") +
  scale_color_manual(values = c("control" = "green", "stressed" = "orange"))

hpc_early_period_model <- lmer(mean_hpc_theta_1_power ~ group * period_type + (1|animal), data=mean_data_early_periods)
summary(hpc_early_period_model)
hpc_early_plot = emmip(hpc_early_period_model, group ~ period_type, CIs = FALSE) + 
  labs(y = "Predicted HPC Power") +
  scale_color_manual(values = c("control" = "green", "stressed" = "orange"))


### Power relationships to each other ###

mean_bla <- mean(mean_data$mean_hpc_theta_1_power, na.rm = TRUE)
sd_bla <- sd(mean_data$mean_hpc_theta_1_power, na.rm = TRUE)
mean_hpc <- mean(mean_data$mean_hpc_theta_1_power, na.rm = TRUE)
sd_hpc <- sd(mean_data$mean_hpc_theta_1_power, na.rm = TRUE)


bla_pl_model <- lmer(mean_pl_theta_1_power ~ group * period_type * mean_bla_theta_1_power + (1|animal), data=mean_data)
summary(bla_pl_model)

bla_pl_plot <- emmip(bla_pl_model, period_type ~ mean_bla_theta_1_power | group,
                     at = list(mean_bla_theta_1_power = c(mean_bla - sd_bla, mean_bla, mean_bla + sd_bla)),
                     CIs = FALSE) + 
  labs(y = "Predicted PL Power") 


hpc_pl_model <- lmer(mean_pl_theta_1_power ~ group * period_type * mean_hpc_theta_1_power + (1|animal), data=mean_data)
summary(hpc_pl_model)



hpc_pl_plot <- emmip(hpc_pl_model, period_type ~ mean_hpc_theta_1_power | group,
                     at = list(mean_hpc_theta_1_power = c(mean_hpc - sd_hpc, mean_hpc, mean_hpc + sd_hpc)),
                     CIs = FALSE) + 
  labs(y = "Predicted PL Power") 








