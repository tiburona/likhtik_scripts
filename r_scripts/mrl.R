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


csv_dir = '/Users/katie/likhtik/IG_INED_Safety_Recall/power'
csv_name = 'mrl_power.csv'
csv_file = paste(csv_dir, csv_name, sep='/')
df <- read.csv(csv_file, comment.char="#") 
df <- subset(df, neuron_quality != '3')



prepare_df <- function(csv_file, frequency_band, brain_region, evoked=FALSE){
 
  df <- read.csv(csv_file, comment.char="#") 
  
  df <- subset(df, neuron_quality != '3')
  
  
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


analyze_data <- function(csv_file, frequency_band, brain_region) {
  data = prepare_df(csv_file, frequency_band, brain_region)
  formula = 'mean_mrl ~ group*neuron_type*period_type + (1|animal/unit)'
  print(brain_region)
  print(frequency_band)
  print(formula)
  model = lmer(formula=formula, data=data)
  plot = emmip(model, group ~ period_type | neuron_type, CIs = FALSE) + 
    labs(y = paste("Predicted", toupper(brain_region), frequency_band, "MRL")) +
    scale_color_manual(values = c("control" = "green", "defeat" = "orange"))
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



### Theta 1 ###

pl_theta_1_result = analyze_data(csv_file, 'theta_1', 'pl')
summary(pl_theta_1_result$model)
pl_theta_1_result$plot
pl_theta_1_bootstrap = bootstrap_model(pl_theta_1_result$model)


### Theta 2 ###

pl_theta_2_result = analyze_data(csv_file, 'theta_2', 'pl')
summary(pl_theta_2_result$model)
pl_theta_2_result$plot
pl_theta_2_bootstrap = bootstrap_model(pl_theta_2_result$model)



#### HPC #####

### Theta 1 ###

hpc_theta_1_result = analyze_data(csv_file, 'theta_1', 'hpc')
summary(hpc_theta_1_result$model)
hpc_theta_1_result$plot
hpc_theta_1_bootstrap = bootstrap_model(hpc_theta_1_result$model)



### Theta 2 ###

hpc_theta_2_result = analyze_data(csv_file, 'theta_2', 'hpc')
summary(hpc_theta_2_result$model)
hpc_theta_2_result$plot
hpc_theta_2_bootstrap = bootstrap_model(hpc_theta_2_result$model)




#### BLA #####


### Theta 1 ###

bla_theta_1_result = analyze_data(csv_file, 'theta_1', 'bla')
summary(bla_theta_1_result$model)
bla_theta_1_result$plot
bla_theta_1_bootstrap = bootstrap_model(bla_theta_1_result$model)



### Theta 2 ###

bla_theta_2_result = analyze_data(csv_file, 'theta_2', 'bla')
summary(bla_theta_2_result$model)
bla_theta_2_result$plot
bla_theta_2_bootstrap = bootstrap_model(bla_theta_2_result$model)



