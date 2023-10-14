
library(glmmTMB)
library(ggpattern)
library(lme4)
library(lmerTest)  # for lmer p-values
library(readr)
library(sjPlot)
library(ggplot2)
library(broom)
library(emmeans)
library(dplyr)
library(rlang)
library(readr)


csv_dir = '/Users/katie/likhtik/data/lfp/psth'

csv_name = 'mrl_pl_theta_1_mrl_pl_theta_2_mrl_bla_theta_1_mrl_bla_theta_2_mrl_hpc_theta_1_mrl_hpc_theta_2_psth_rsGlX0.csv'

csv_file = paste(csv_dir, csv_name, sep='/')
df <- read.csv(csv_file, comment.char="#") 

factor_vars <- c('animal', 'group', 'neuron_type', 'period_type', 'unit')
df[factor_vars] <- lapply(df[factor_vars], factor)

bla_theta_1_firing_rate_model <- lmer(rate ~ bla_theta_1_mrl * group * period_type * neuron_type + (1|animal/unit), data=df)
summary(bla_theta_1_firing_rate_model)

# Compute the EMMs

mn = mean(df$bla_theta_1_mrl, na.rm=TRUE)
sd = sd(df$bla_theta_1_mrl, na.rm=TRUE)
below = mn - sd
above = mn + sd

# Create the emmip graph
emmip_plot <- emmip(bla_theta_1_firing_rate_model, period_type ~ bla_theta_1_mrl | neuron_type, 
                    at = list(bla_theta_1_mrl = c(below, mn, above)))

print(emmip_plot)



csv_name = 'mrl_pl_theta_1_mrl_pl_theta_2_mrl_bla_theta_1_mrl_bla_theta_2_mrl_hpc_theta_1_mrl_hpc_theta_2_psth_VI3DKQ.csv'

csv_name = 'mrl_pl_theta_1_mrl_pl_theta_2_mrl_bla_theta_1_mrl_bla_theta_2_mrl_hpc_theta_1_mrl_hpc_theta_2_psth_percent_freezing_5IBgEv.csv'
csv_name = 'mrl_pl_theta_1_mrl_pl_theta_2_mrl_bla_theta_1_mrl_bla_theta_2_mrl_hpc_theta_1_mrl_hpc_theta_2_psth_percent_freezing_s6mP1n.csv'

behavior_firing_rate_model <- lmer(rate ~ percent_freezing * group * period_type * neuron_type + (1|animal/unit), data=df)
summary(behavior_firing_rate_model)

new_data <- expand.grid(
  percent_freezing = c(mean(df$percent_freezing) - sd(df$percent_freezing), 
                       mean(df$percent_freezing), 
                       mean(df$percent_freezing) + sd(df$percent_freezing)),
  group = unique(df$group),
  period_type = unique(df$period_type),
  neuron_type = unique(df$neuron_type)
)

new_data$predicted_rate <- predict(behavior_firing_rate_model, newdata = new_data, re.form = NA)

ggplot(new_data, aes(x = percent_freezing, y = predicted_rate, color = period_type)) +
  geom_line() +
  labs(x = "Percent Freezing", y = "Predicted Firing Rate") +
  theme_bw() +
  facet_grid(neuron_type ~ group, scales = "free")

csv_name = 'mrl_pl_theta_1_mrl_pl_theta_2_mrl_bla_theta_1_mrl_bla_theta_2_mrl_hpc_theta_1_mrl_hpc_theta_2_psth_percent_freezing__lfp_from_prev_period.csv'

csv_name = 'psth_mrl_pl_theta_1_mrl_pl_theta_2_mrl_bla_theta_1_mrl_bla_theta_2_mrl_hpc_theta_1_mrl_hpc_theta_2_percent_freezing__lfp_from_prev_period_row_is_trial.csv'

bla_theta_1_firing_rate_model_trials <- lmer(rate ~ bla_theta_1_mrl * group * period_type * neuron_type + (1|animal/unit/period), data=df)
summary(bla_theta_1_firing_rate_model_trials)


# Create the emmip graph
emmip_plot <- emmip(bla_theta_1_firing_rate_model_trials, period_type ~ bla_theta_1_mrl | neuron_type, 
                    at = list(bla_theta_1_mrl = c(below, mn, above)))

print(emmip_plot)


new_data <- expand.grid(
  bla_theta_1_mrl = c(mean(df$bla_theta_1_mrl, na.rm=TRUE) - sd(df$bla_theta_1_mrl, na.rm=TRUE), 
                      mean(df$bla_theta_1_mrl, na.rm=TRUE), 
                      mean(df$bla_theta_1_mrl, na.rm=TRUE) + sd(df$bla_theta_1_mrl, na.rm=TRUE)),
  
  group = unique(df$group),
  period_type = unique(df$period_type),
  neuron_type = unique(df$neuron_type)
)

new_data$predicted_rate <- predict(bla_theta_1_firing_rate_model, newdata = new_data, re.form = NA)

ggplot(new_data, aes(x = bla_theta_1_mrl, y = predicted_rate, color = period_type)) +
  geom_line() +
  labs(x = "BLA Theta 1 MRL", y = "Predicted Firing Rate") +
  theme_bw() +
  facet_grid(neuron_type ~ group, scales = "free")

bla_theta_1_firing_rate_model <- lmer(rate ~ bla_theta_1_mrl * group * period_type * neuron_type + (1|animal/unit), data=df)

pl_theta_1_firing_rate_model <- lmer(rate ~ pl_theta_1_mrl * group * period_type * neuron_type + (1|animal/unit), data=df)
summary(pl_theta_1_firing_rate_model)

new_data_pl <- expand.grid(
  pl_theta_1_mrl = c(mean(df$pl_theta_1_mrl, na.rm=TRUE) - sd(df$pl_theta_1_mrl, na.rm=TRUE), 
                      mean(df$pl_theta_1_mrl, na.rm=TRUE), 
                      mean(df$pl_theta_1_mrl, na.rm=TRUE) + sd(df$pl_theta_1_mrl, na.rm=TRUE)),
  
  group = unique(df$group),
  period_type = unique(df$period_type),
  neuron_type = unique(df$neuron_type)
)

new_data_pl$predicted_rate <- predict(pl_theta_1_firing_rate_model, newdata = new_data_pl, re.form = NA)

ggplot(new_data_pl, aes(x = pl_theta_1_mrl, y = predicted_rate, color = period_type)) +
  geom_line() +
  labs(x = "PL Theta 1 MRL", y = "Predicted Firing Rate") +
  theme_bw() +
  facet_grid(neuron_type ~ group, scales = "free")


hpc_theta_1_firing_rate_model <- lmer(rate ~ hpc_theta_1_mrl * group * period_type * neuron_type + (1|animal/unit), data=df)
summary(hpc_theta_1_firing_rate_model)

mn = mean(df$hpc_theta_1_mrl, na.rm=TRUE)
sd = sd(df$hpc_theta_1_mrl, na.rm=TRUE)
below = mn - sd
above = mn + sd
emmip_plot <- emmip(hpc_theta_1_firing_rate_model, period_type ~ hpc_theta_1_mrl | neuron_type, 
                    at = list(hpc_theta_1_mrl = c(below, mn, above)))

print(emmip_plot)
