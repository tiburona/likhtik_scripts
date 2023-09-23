
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
library(brms)
library(data.table)

csv_dir = '/Users/katie/likhtik/data/lfp/psth'

csv_name = 'mrl_pl_theta_1_mrl_pl_theta_2_mrl_bla_theta_1_mrl_bla_theta_2_mrl_hpc_theta_1_mrl_hpc_theta_2_psth_68G6Lq.csv'

csv_file = paste(csv_dir, csv_name, sep='/')
df <- read.csv(csv_file, comment.char="#") 

factor_vars <- c('animal', 'group', 'neuron_type', 'unit',)
df[factor_vars] <- lapply(df[factor_vars], factor)

bla_theta_1_firing_rate_model <- lmer(rate ~ bla_theta_1_mrl * group * neuron_type + (1|animal/unit), data=df)
summary(bla_theta_1_firing_rate_model)

# Compute the EMMs
emm_res <- emmeans(bla_theta_1_firing_rate_model, ~ bla_theta_1_mrl * group * period_type * neuron_type)

# Create the emmip graph
emmip_plot <- emmip(emm_res, period_type ~ group * bla_theta_1_mrl | neuron_type, position = "dodge")

print(emmip_plot)

# Convert dataframe to data table
dt <- as.data.table(df)

# Define the columns by which to split
split_cols <- setdiff(names(dt), c("rate", "pl_theta_1_mrl", "period_type"))

# Group by the split columns and create a list of sub-data.tables
splits_list <- dt[, .SD, by = split_cols]

splits <- split(splits_list, seq_len(nrow(splits_list)))


compute_diff <- function(subdata) {
  if (nrow(subdata) != 2) {
    return(NULL)  # Skip subsets that don't have both tone and pretone
  }
  
  tone_data <- subdata[subdata$period_type == "tone", ]
  pretone_data <- subdata[subdata$period_type == "pretone", ]
  
  if (nrow(tone_data) == 0 || nrow(pretone_data) == 0) {
    return(NULL)  # Skip if either tone or pretone data is missing
  }
  
  evoked_pl_theta_1_mrl <- tone_data$pl_theta_1_mrl - pretone_data$pl_theta_1_mrl
  evoked_rate <- tone_data$rate - pretone_data$rate
  
  # Return a dataframe with the differences and the other columns (excluding rate, pl_theta_1_mrl, and period_type)
  return(data.frame(subdata[1, setdiff(names(subdata), c("rate", "pl_theta_1_mrl", "period_type"))], evoked_pl_theta_1_mrl, evoked_rate))
}

results_list <- lapply(splits, compute_diff)
results_list <- Filter(Negate(is.null), results_list)  # Remove NULLs
result <- do.call(rbind, results_list)



