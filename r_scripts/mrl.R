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


csv_dir = '/Users/katie/likhtik/data/lfp/percent_freezing'



prepare_df <- function(frequency_band, brain_region, evoked=FALSE){
 
  csv_name = 'spike_power_mrl.csv'
  csv_file = paste(csv_dir, csv_name, sep='/')
  df <- read.csv(csv_file, comment.char="#") 
  
  # Convert variables to factors
  factor_vars <- c('animal', 'group', 'period_type', 'neuron_type', 'unit')
  df[factor_vars] <- lapply(df[factor_vars], factor)
  
  # Convert frequency_band to a symbol for tidy evaluation
  mrl_column <- sym(paste0(brain_region, '_', frequency_band, "_mrl"))
  
  averaged_result <- df %>%
    group_by(animal, period_type, period, neuron_type, group, unit) %>%
    summarize(
      mean_mrl = mean(!!mrl_column, na.rm = TRUE)
    ) %>%
    ungroup()
  
  data_to_return <- averaged_result
  
  return(data_to_return) 
}


analyze_data <- function(frequency_band, brain_region) {
  data = prepare_df(frequency_band, brain_region)
  formula = 'mean_mrl ~ group*neuron_type*period_type + (1|animal/unit)'
  model = lmer(formula=formula, data=data)
  plot = emmip(model, group ~ period_type | neuron_type, CIs = FALSE)
  return(list(model = model, plot = plot, data=data))
}


bootstrap_model <- function(fit, nsim=1000) {
  boot_results <- bootMer(fit, FUN=fixef, nsim)
  
  num_effects <- ncol(boot_results$t)
  
  # Initialize a matrix to store the confidence intervals:
  ci_matrix <- matrix(0, nrow=2, ncol=num_effects)
  
  # Loop through each effect to calculate confidence intervals:
  for (i in 1:num_effects) {
    ci_matrix[,i] <- quantile(boot_results$t[,i], c(0.025, 0.975))
  }
  
  # Convert matrix to data frame and add names for clarity:
  ci_df <- as.data.frame(t(ci_matrix))
  colnames(ci_df) <- c("2.5%", "97.5%")
  rownames(ci_df) <- names(fixef(fit))
  
  # Add an asterisk for effects where CI does not include 0:
  ci_df$significance <- ifelse(ci_df$`2.5%` > 0 | ci_df$`97.5%` < 0, "*", "")
  
  # Print the results:
  print(ci_df)
  
  # Return the data frame in case the user wants to save or manipulate it further:
  return(ci_df)
}



#### PL #####
### Delta ###

pl_delta_result = analyze_data('delta', 'pl')
summary(pl_delta_result$model)
pl_delta_result$plot
pl_delta_bootstrap = bootstrap_model(pl_delta_result$model)

pl_delta_evoked_result = analyze_data('delta', 'pl', evoked=TRUE)
summary(pl_delta_evoked_result$model)
pl_delta_evoked_result$plot
pl_delta_evoked_bootstrap = bootstrap_model(pl_delta_evoked_result$model)



### Theta 1 ###

pl_theta_1_result = analyze_data('theta_1', 'pl')
summary(pl_theta_1_result$model)
pl_theta_1_result$plot
pl_theta_1_bootstrap = bootstrap_model(pl_theta_1_result$model)

pl_theta_1_evoked_result = analyze_data('theta_1', 'pl', evoked=TRUE)
summary(pl_theta_1_evoked_result$model)
pl_theta_1_evoked_result$plot
pl_theta_1_evoked_bootstrap = bootstrap_model(pl_theta_1_evoked_result$model)



### Theta 2 ###

pl_theta_2_result = analyze_data('theta_2', 'pl')
summary(pl_theta_2_result$model)
pl_theta_2_result$plot
pl_theta_2_bootstrap = bootstrap_model(pl_theta_2_result$model)

pl_theta_2_evoked_result = analyze_data('theta_2', 'pl', evoked=TRUE)
summary(pl_theta_2_evoked_result$model)
pl_theta_2_evoked_result$plot
pl_theta_2_evoked_bootstrap = bootstrap_model(pl_theta_2_evoked_result$model)


### Gamma ###

pl_gamma_result = analyze_data('gamma', 'pl')
summary(pl_gamma_result$model)
pl_gamma_result$plot
pl_gamma_bootstrap = bootstrap_model(pl_gamma_result$model)

pl_gamma_evoked_result = analyze_data('gamma', 'pl', evoked=TRUE)
summary(pl_gamma_evoked_result$model)
pl_gamma_evoked_result$plot
pl_gamma_evoked_bootstrap = bootstrap_model(pl_gamma_evoked_result$model)


### HGamma ###

pl_hgamma_result = analyze_data('hgamma', 'pl')
summary(pl_hgamma_result$model)
pl_hgamma_result$plot
pl_hgamma_bootstrap = bootstrap_model(pl_hgamma_result$model)

pl_hgamma_evoked_result = analyze_data('hgamma', 'pl', evoked=TRUE)
summary(pl_hgamma_evoked_result$model)
pl_hgamma_evoked_result$plot
pl_hgamma_evoked_bootstrap = bootstrap_model(pl_hgamma_evoked_result$model)




#### HPC #####
### Delta ###

hpc_delta_result = analyze_data('delta', 'hpc')
summary(hpc_delta_result$model)
hpc_delta_result$plot
hpc_delta_bootstrap = bootstrap_model(hpc_delta_result$model)

hpc_delta_evoked_result = analyze_data('delta', 'hpc', evoked=TRUE)
summary(hpc_delta_evoked_result$model)
hpc_delta_evoked_result$plot
hpc_delta_evoked_bootstrap = bootstrap_model(hpc_delta_evoked_result$model)



### Theta 1 ###

hpc_theta_1_result = analyze_data('theta_1', 'hpc')
summary(hpc_theta_1_result$model)
hpc_theta_1_result$plot
hpc_theta_1_bootstrap = bootstrap_model(hpc_theta_1_result$model)

hpc_theta_1_evoked_result = analyze_data('theta_1', 'hpc', evoked=TRUE)
summary(hpc_theta_1_evoked_result$model)
hpc_theta_1_evoked_result$plot
hpc_theta_1_evoked_bootstrap = bootstrap_model(hpc_theta_1_evoked_result$model)



### Theta 2 ###

hpc_theta_2_result = analyze_data('theta_2', 'hpc')
summary(hpc_theta_2_result$model)
hpc_theta_2_result$plot
hpc_theta_2_bootstrap = bootstrap_model(hpc_theta_2_result$model)

hpc_theta_2_evoked_result = analyze_data('theta_2', 'hpc', evoked=TRUE)
summary(hpc_theta_2_evoked_result$model)
hpc_theta_2_evoked_result$plot
hpc_theta_2_evoked_bootstrap = bootstrap_model(hpc_theta_2_evoked_result$model)


### Gamma ###

hpc_gamma_result = analyze_data('gamma', 'hpc')
summary(hpc_gamma_result$model)
hpc_gamma_result$plot
hpc_gamma_bootstrap = bootstrap_model(hpc_gamma_result$model)

hpc_gamma_evoked_result = analyze_data('gamma', 'hpc', evoked=TRUE)
summary(hpc_gamma_evoked_result$model)
hpc_gamma_evoked_result$plot
hpc_gamma_evoked_bootstrap = bootstrap_model(hpc_gamma_evoked_result$model)


### HGamma ###

hpc_hgamma_result = analyze_data('hgamma', 'hpc')
summary(hpc_hgamma_result$model)
hpc_hgamma_result$plot
hpc_hgamma_bootstrap = bootstrap_model(hpc_hgamma_result$model)

hpc_hgamma_evoked_result = analyze_data('hgamma', 'hpc', evoked=TRUE)
summary(hpc_hgamma_evoked_result$model)
hpc_hgamma_evoked_result$plot
hpc_hgamma_evoked_bootstrap = bootstrap_model(hpc_hgamma_evoked_result$model)


#### BLA #####
### Delta ###

bla_delta_result = analyze_data('delta', 'bla')
summary(bla_delta_result$model)
bla_delta_result$plot
bla_delta_bootstrap = bootstrap_model(bla_delta_result$model)

bla_delta_evoked_result = analyze_data('delta', 'bla', evoked=TRUE)
summary(bla_delta_evoked_result$model)
bla_delta_evoked_result$plot
bla_delta_evoked_bootstrap = bootstrap_model(bla_delta_evoked_result$model)



### Theta 1 ###

bla_theta_1_result = analyze_data('theta_1', 'bla')
summary(bla_theta_1_result$model)
bla_theta_1_result$plot
bla_theta_1_bootstrap = bootstrap_model(bla_theta_1_result$model)

bla_theta_1_evoked_result = analyze_data('theta_1', 'bla', evoked=TRUE)
summary(bla_theta_1_evoked_result$model)
bla_theta_1_evoked_result$plot
bla_theta_1_evoked_bootstrap = bootstrap_model(bla_theta_1_evoked_result$model)


### Theta 2 ###

bla_theta_2_result = analyze_data('theta_2', 'bla')
summary(bla_theta_2_result$model)
bla_theta_2_result$plot
bla_theta_2_bootstrap = bootstrap_model(bla_theta_2_result$model)

bla_theta_2_evoked_result = analyze_data('theta_2', 'bla', evoked=TRUE)
summary(bla_theta_2_evoked_result$model)
bla_theta_2_evoked_result$plot
bla_theta_2_evoked_bootstrap = bootstrap_model(bla_theta_2_evoked_result$model)


### Gamma ###

bla_gamma_result = analyze_data('gamma', 'bla')
summary(bla_gamma_result$model)
bla_gamma_result$plot
bla_gamma_bootstrap = bootstrap_model(bla_gamma_result$model)

bla_gamma_evoked_result = analyze_data('gamma', 'bla', evoked=TRUE)
summary(bla_gamma_evoked_result$model)
bla_gamma_evoked_result$plot
bla_gamma_evoked_bootstrap = bootstrap_model(bla_gamma_evoked_result$model)


### HGamma ###

bla_hgamma_result = analyze_data('hgamma', 'bla')
summary(bla_hgamma_result$model)
bla_hgamma_result$plot
bla_hgamma_bootstrap = bootstrap_model(bla_hgamma_result$model)

bla_hgamma_evoked_result = analyze_data('hgamma', 'bla', evoked=TRUE)
summary(bla_hgamma_evoked_result$model)
bla_hgamma_evoked_result$plot
bla_hgamma_evoked_bootstrap = bootstrap_model(bla_hgamma_evoked_result$model)