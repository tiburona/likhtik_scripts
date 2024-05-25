library(lme4)
library(lmerTest)
library(emmeans)
library(ggplot2)
library(dplyr)


# Load the CSV file
count_data <- 
  read.csv('/Users/katie/likhtik/IG_INED_Safety_Recall/spike_counts/count_spreadsheet.csv', 
           comment.char="#")


count_data <- subset(count_data, count_data$time_bin < 30)

count_data <- count_data %>%
  group_by(group, neuron_type, period_type, animal, unit, period, event) %>%
  summarise(count = sum(spike_counts, na.rm = TRUE), .groups = 'drop')


# Make sure every categorical variable is a factor
factor_vars <- c('animal', 'group', 'period_type', 'neuron_type', 'unit')
hmm_demo_data[factor_vars] <- lapply(hmm_demo_data[factor_vars], factor)
