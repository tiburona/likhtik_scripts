library(glmmTMB)
library(lme4)
library(lmerTest)  # for lmer p-values
library(readr)
library(sjPlot)
library(ggplot2)
library(broom)
library(emmeans)
library(dplyr)

csv_file = '/Users/katie/likhtik/data/mrl_theta_1_period_frequency_periods.csv'



prepare_df <- function(csv){
  df <- read_csv(csv)
  
  # Convert variables to factors
  factor_vars <- c('animal', 'group', 'period_type', 'neuron_type', 'unit')
  df[factor_vars] <- lapply(df[factor_vars], factor)
  
  return(df)
}


data = prepare_df(csv_file)

theta1_model <- lmer(theta_1_mrl ~ group*neuron_type*period_type + (1|animal/unit), data=data)
summary(theta1_model)

plot <- emmip(theta1, group ~ period_type | neuron_type, CIs = FALSE)
