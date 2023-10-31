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
library(afex)

process_data <- function(frequency_band, brain_region) {
  
  csv_name = sprintf('mrl_%s_continuous_period_frequency_bins_%s.csv', frequency_band, brain_region)
  csv_file = paste(csv_dir, csv_name, sep='/')
  df <- read_csv(csv_file) 
  
  # Step 1: Rename the variable
  mrl_var <- grep("_mrl$", names(df), value = TRUE)
  names(df)[names(df) == mrl_var] <- "mrl"
  
  # Step 2: Subset the data
  df <- df[df$period %in% c(0, 1), ]
  
  # Step 3: Average frequency_bin and period grouping by all other variables
  df_avg <- df %>%
    group_by(group, neuron_type, period_type, animal, unit) %>%
    summarise(mrl = mean(mrl, na.rm = TRUE))
  
  # Step 4: Subtract pretone from tone values
  df_pretone <- df_avg[df_avg$period_type == 'pretone', ]
  df_tone <- df_avg[df_avg$period_type == 'tone', ]
  
  df_sub <- merge(df_pretone, df_tone, by = c("group", "neuron_type", "animal", "unit"))
  df_sub$mrl_diff <- df_sub$mrl.y - df_sub$mrl.x
  
  # Return the resulting data frame
  return(df_sub)
}


get_aov_res <- function(data) {
  # Create an interaction term for animal and unit
  data$animal_unit <- interaction(data$animal, data$unit)
  
  # Fit the mixed-design ANOVA
  model <- aov(mrl_diff ~ group * neuron_type + Error(animal_unit/neuron_type), data = result)
  
  # Print the summary
  summary(model)
  
  return(model)
}


#### PL #####
### Delta ###


pl_delta_result = analyze_data('delta', 'pl')
summary(pl_delta_result$model)
pl_delta_result$plot
pl_delta_bootstrap = bootstrap_model(pl_delta_result$model)


### Theta 1 ###

pl_theta_1_result = analyze_data('theta_1', 'pl')
summary(pl_theta_1_result$model)
pl_theta_1_result$plot
pl_theta_1_bootstrap = bootstrap_model(pl_theta_1_result$model)


### Theta 2 ###

pl_theta_2_result = analyze_data('theta_2', 'pl')
summary(pl_theta_2_result$model)
pl_theta_2_result$plot
pl_theta_2_bootstrap = bootstrap_model(pl_theta_2_result$model)



### Gamma ###

pl_gamma_result = analyze_data('gamma', 'pl')
summary(pl_gamma_result$model)
pl_gamma_result$plot
pl_gamma_bootstrap = bootstrap_model(pl_gamma_result$model)


### HGamma ###

pl_hgamma_result = analyze_data('hgamma', 'pl')
summary(pl_hgamma_result$model)
pl_hgamma_result$plot
pl_hgamma_bootstrap = bootstrap_model(pl_hgamma_result$model)





#### HPC #####
### Delta ###

hpc_delta_result = analyze_data('delta', 'hpc')
summary(hpc_delta_result$model)
hpc_delta_result$plot
hpc_delta_bootstrap = bootstrap_model(hpc_delta_result$model)





### Theta 1 ###

hpc_theta_1_result = analyze_data('theta_1', 'hpc')
summary(hpc_theta_1_result$model)
hpc_theta_1_result$plot
hpc_theta_1_bootstrap = bootstrap_model(hpc_theta_1_result$model)





### Theta 2 ###

hpc_theta_2_result = analyze_data('theta_2', 'hpc')
summary(hpc_theta_2_result$model)
hpc_theta_2_result$plot
hpc_theta_2_bootstrap = bootstrap_model(hpc_theta_2_result$model)




### Gamma ###

hpc_gamma_result = analyze_data('gamma', 'hpc')
summary(hpc_gamma_result$model)
hpc_gamma_result$plot
hpc_gamma_bootstrap = bootstrap_model(hpc_gamma_result$model)




### HGamma ###

hpc_hgamma_result = analyze_data('hgamma', 'hpc')
summary(hpc_hgamma_result$model)
hpc_hgamma_result$plot
hpc_hgamma_bootstrap = bootstrap_model(hpc_hgamma_result$model)




#### BLA #####
### Delta ###

bla_delta_result = analyze_data('delta', 'bla')
summary(bla_delta_result$model) #trend level result
bla_delta_result$plot
bla_delta_bootstrap = bootstrap_model(bla_delta_result$model)





### Theta 1 ###

bla_theta_1_result = analyze_data('theta_1', 'bla')
summary(bla_theta_1_result$model)
bla_theta_1_result$plot
bla_theta_1_bootstrap = bootstrap_model(bla_theta_1_result$model)




### Theta 2 ###

bla_theta_2_result = analyze_data('theta_2', 'bla')
summary(bla_theta_2_result$model)
bla_theta_2_result$plot
bla_theta_2_bootstrap = bootstrap_model(bla_theta_2_result$model)



### Gamma ###

bla_gamma_result = analyze_data('gamma', 'bla')
summary(bla_gamma_result$model)
bla_gamma_result$plot
bla_gamma_bootstrap = bootstrap_model(bla_gamma_result$model)




### HGamma ###

bla_hgamma_result = analyze_data('hgamma', 'bla')
summary(bla_hgamma_result$model)
bla_hgamma_result$plot
bla_hgamma_bootstrap = bootstrap_model(bla_hgamma_result$model)