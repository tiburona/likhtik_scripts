library(lme4)
library(dplyr)
library(ggplot2)
library(emmeans)
library(rlang)
library(lmerTest)




csv_dir = '/Users/katie/likhtik/IG_INED_Safety_Recall/mrl'

csv_name = 'previous_mrl_firing_rate.csv'

csv_file = paste(csv_dir, csv_name, sep='/')
prev_mrl_rate_df <- read.csv(csv_file, comment.char="#") 


read_metadata(csv_file)

prev_mrl_rate_df$neuron_type <- factor(prev_mrl_rate_df$neuron_type,
                                  levels = c("IN", "PN"))
prev_mrl_rate_df <- mrl_rate_df[!is.na(prev_mrl_rate_df[['neuron_type']]), ]


factor_vars <- c('animal', 'group', 'period_type', 'unit')
prev_mrl_rate_df[factor_vars] <- lapply(mrl_rate_df[factor_vars], factor)

prev_mrl_rate_data <- prev_mrl_rate_df %>%
  filter(time_bin < 5) %>%
  group_by(period, group, period_type, neuron_type, unit, animal, event) %>%
  summarise(
    rate = mean(rate, na.rm = TRUE),
    hpc_theta_1_mrl = mean(hpc_theta_1_mrl, na.rm = TRUE),
    bla_theta_1_mrl = mean(bla_theta_1_mrl, na.rm = TRUE),
    pl_theta_1_mrl  = mean(pl_theta_1_mrl, na.rm = TRUE),
    pl_theta_2_mrl = mean(pl_theta_2_mrl, na.rm = TRUE),
    bla_theta_2_mrl = mean(bla_theta_2_mrl, na.rm = TRUE),
    hpc_theta_2_mrl = mean(hpc_theta_2_mrl, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )


clean_data <- prev_mrl_rate_data[!is.na(prev_mrl_rate_data[['bla_theta_1_mrl']]), ]
bla_theta_1_mrl_rate <- lmer(rate ~ bla_theta_1_mrl * neuron_type* period_type * group + (1|animal:unit) + (1|animal:unit:period), data = clean_data)
summary(bla_theta_1_mrl_rate)
pred_data = create_predictions_data(clean_data, bla_theta_1_mrl_rate, 'bla_theta_1_mrl')
plot <- graph_predictions(pred_data, 'bla_theta_1_mrl', "predicted", 'bla_theta_1_mrl', paste("Predicted", 'Rate'), num_vars=4)

