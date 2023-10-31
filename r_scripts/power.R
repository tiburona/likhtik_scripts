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

# Read in all lines of the file
all_lines <- readLines(deviation_csv)

# Filter lines that start with the comment character
metadata_lines <- all_lines[grepl("^#", all_lines)]

# Print the metadata lines
cat(metadata_lines, sep = "\n")



mean_data <- deviation_data %>%
  filter(time_bin < 30) %>%
  group_by(period, group, period_type) %>%
  summarise(
    mean_hpc_theta_1_power = mean(hpc_theta_1_power, na.rm = TRUE),
    mean_bla_theta_1_power = mean(bla_theta_1_power, na.rm = TRUE),
    mean_pl_theta_1_power  = mean(pl_theta_1_power, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )

mean_data_early_periods <- mean_data %>%
  filter(period < 2)


# BLA #
ggplot(mean_data, aes(x = period, y = mean_bla_theta_1_power, color = period_type, shape = period_type)) +
  stat_summary(fun = mean, geom = "point", size = 3, aes(shape = period_type, color = period_type)) +
  stat_summary(fun = mean, geom = "line", aes(group = period_type)) +
  scale_shape_manual(values = c("pretone" = 15, "tone" = 16)) + # 15 is for squares and 16 is for circles
  scale_color_manual(values = c("pretone" = "gray", "tone" = "blue")) +
  labs(x = "Period", y = "Mean BLA Theta 1 Power") +
  facet_wrap(~ group, ncol = 1) + # Separate panels for each level of 'group'
  theme_minimal()



# PL #

ggplot(mean_data, aes(x = period, y = mean_pl_theta_1_power, color = period_type, shape = period_type)) +
  stat_summary(fun = mean, geom = "point", size = 3, aes(shape = period_type, color = period_type)) +
  stat_summary(fun = mean, geom = "line", aes(group = period_type)) +
  scale_shape_manual(values = c("pretone" = 15, "tone" = 16)) + # 15 is for squares and 16 is for circles
  scale_color_manual(values = c("pretone" = "gray", "tone" = "blue")) +
  labs(x = "Period", y = "Mean PL Theta 1 Power") +
  facet_wrap(~ group, ncol = 1) + # Separate panels for each level of 'group'
  theme_minimal()

# HPC #

ggplot(mean_data, aes(x = period, y = mean_hpc_theta_1_power, color = period_type, shape = period_type)) +
  stat_summary(fun = mean, geom = "point", size = 3, aes(shape = period_type, color = period_type)) +
  stat_summary(fun = mean, geom = "line", aes(group = period_type)) +
  scale_shape_manual(values = c("pretone" = 15, "tone" = 16)) + # 15 is for squares and 16 is for circles
  scale_color_manual(values = c("pretone" = "gray", "tone" = "blue")) +
  labs(x = "Period", y = "Mean HPC Theta 1 Power") +
  facet_wrap(~ group, ncol = 1) + # Separate panels for each level of 'group'
  theme_minimal()


### Models ####

mean_data <- deviation_data %>%
  filter(time_bin < 30) %>%
  group_by(period, group, period_type, animal, ) %>%
  summarise(
    mean_hpc_theta_1_power = mean(hpc_theta_1_power, na.rm = TRUE),
    mean_bla_theta_1_power = mean(bla_theta_1_power, na.rm = TRUE),
    mean_pl_theta_1_power  = mean(pl_theta_1_power, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )

mean_data_early_periods <- mean_data %>%
  filter(period < 2)



bla_model <- lmer(mean_bla_theta_1_power ~ group * period_type + (1|animal), data=mean_data)
summary(bla_model)
bla_plot = emmip(bla_model, group ~ period_type, CIs = FALSE) + 
  labs(y = "Predicted BLA Power") +
  scale_color_manual(values = c("control" = "green", "stressed" = "orange"))

bla_early_period_model <- lmer(mean_bla_theta_1_power ~ group * period_type + (1|animal), data=mean_data_early_periods)
summary(bla_early_period_model)
bla_early_plot = emmip(bla_early_period_model, group ~ period_type, CIs = FALSE) + 
  labs(y = "Predicted PL Power") +
  scale_color_manual(values = c("control" = "green", "stressed" = "orange"))


pl_model <- lmer(mean_pl_theta_1_power ~ group * period_type + (1|animal), data=mean_data)
summary(pl_model)
pl_plot = emmip(pl_model, group ~ period_type, CIs = FALSE) + 
  labs(y = "Predicted PL Power") +
  scale_color_manual(values = c("control" = "green", "stressed" = "orange"))

pl_early_period_model <- lmer(mean_pl_theta_1_power ~ group * period_type + (1|animal), data=mean_data_early_periods)
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








