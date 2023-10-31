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


mrl_rate_data <- mrl_rate_df %>%
  filter(time_bin < 30) %>%
  group_by(period, group, period_type, neuron_type, unit, animal) %>%
  summarise(
    rate = mean(rate, na.rm = TRUE),
    hpc_theta_1_mrl = mean(hpc_theta_1_mrl, na.rm = TRUE),
    bla_theta_1_mrl = mean(bla_theta_1_mrl, na.rm = TRUE),
    pl_theta_1_mrl  = mean(pl_theta_1_mrl, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )

group_colors = c("green", "orange")
make_mrl_plot <- function(model) {
  plot <- emmip(model, group ~ period_type | neuron_type, CIs = FALSE) +
    scale_color_manual(values = c("green", "orange"))
  return(plot)
}



# BLA #

bla_model = lmer(bla_theta_1_mrl ~ group * period_type * neuron_type + (1|animal/unit), data=mrl_rate_data)
summary(bla_model)
bla_plot = make_mrl_plot(bla_model) + labs(y = "Predicted BLA MRL") 



# PL #

pl_model = lmer(pl_theta_1_mrl ~ group * period_type * neuron_type + (1|animal/unit), data=mrl_rate_data)
summary(pl_model)
pl_plot =  make_mrl_plot(pl_model) + labs(y = "Predicted PL MRL") 



# HPC #

hpc_model = lmer(hpc_theta_1_mrl ~ group * neuron_type* period_type + (1|animal/unit), data=mrl_rate_data)
summary(hpc_model)
hpc_plot =  make_mrl_plot(hpc_model) + labs(y = "Predicted HPC MRL") 



### MRL relationships to each other ###

bla_pl_model <- lmer(pl_theta_1_mrl ~ group * period_type * bla_theta_1_mrl + (1|animal/unit), data=mrl_rate_data)
summary(bla_pl_model)

bla_pl_predictions <- create_predictions_data(mrl_rate_data, bla_pl_model, 'bla_theta_1_mrl', num_vars=3)
bla_pl_plot <- graph_predictions(bla_pl_predictions, 'bla_theta_1_mrl', 'predicted', 'BLA Theta MRL', 'Predicted HPC MRL Theta', num_vars=3)


bla_hpc_model <- lmer(hpc_theta_1_mrl ~ group * period_type *  bla_theta_1_mrl + (1|animal/unit), data=mrl_rate_data)
summary(bla_hpc_model)

bla_hpc_predictions <- create_predictions_data(mrl_rate_data, bla_hpc_model, 'bla_theta_1_mrl', num_vars=3)
bla_hpc_plot <- graph_predictions(bla_hpc_predictions, 'bla_theta_1_mrl', 'predicted', 'BLA Theta MRL', 'Predicted HPC MRL Theta', num_vars=3)



pl_hpc_model <- lmer(hpc_theta_1_mrl ~ group * period_type *  pl_theta_1_mrl + (1|animal/unit), data=mrl_rate_data)
summary(pl_hpc_model)

pl_hpc_predictions <- create_predictions_data(mrl_rate_data, pl_hpc_model, 'pl_theta_1_mrl', num_vars=3)
pl_hpc_plot <- graph_predictions(pl_hpc_predictions, 'pl_theta_1_mrl', 'predicted', 'PL Theta MRL', 'Predicted HPC MRL Theta', num_vars=3)
