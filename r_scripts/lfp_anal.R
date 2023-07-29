library(glmmTMB)
library(lme4)
library(lmerTest)  # for lmer p-values
library(readr)
library(sjPlot)
library(ggplot2)
library(broom)
library(emmeans)

prepare_df <- function(csv, data_type, observation_type){
  df <- read_csv(csv)
  
  # Convert variables to factors
  factor_vars <- c('unit_num', 'animal', 'category', 'condition')
  df[factor_vars] <- lapply(df[factor_vars], factor)
  
  if (data_type == 'proportion' && observation_type == 'unit') {
    df$"proportion"[df$"proportion" == 0] <- df$"proportion"[df$"proportion" == 0] + 1e-6
  }
  
  
  return(df)
}

theta1_data = prepare_df('/Users/katie/likhtik/data/lfp_psth_theta_2_period_units.csv', 'psth', 'unit')

model = lmer(rate~condition*category*power*period + (1|animal), data=theta1_data)

period1_data = subset(theta1_data, theta1_data$period==2)
