library(glmmTMB)
library(lme4)
library(lmerTest)  # for lmer p-values
library(readr)
library(sjPlot)
library(ggplot2)
library(broom)
library(emmeans)

csv_file = '/Users/katie/likhtik/data/lfp/lfp_theta_1_lfp_theta_2_lfp_delta_lfp_pl_period_units.csv'

# Slide 20

prepare_df <- function(csv){
  df <- read_csv(csv)
  
  # Convert variables to factors
  factor_vars <- c('unit', 'animal', 'category', 'condition', 'period_type')
  df[factor_vars] <- lapply(df[factor_vars], factor)
  
  return(df)
}

df = prepare_df(csv_file)
