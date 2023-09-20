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


csv_dir = '/Users/katie/likhtik/data/lfp/mrl'



prepare_df <- function(frequency_band, brain_region, average_unit=FALSE){
  
  csv_name = sprintf('mrl_%s_continuous_period_frequency_bins_wavelet_%s.csv', frequency_band, brain_region)
  csv_file = paste(csv_dir, csv_name, sep='/')
  df <- read_csv(csv_file) 
  
  # Convert variables to factors
  factor_vars <- c('animal', 'group', 'period_type', 'neuron_type', 'unit')
  df[factor_vars] <- lapply(df[factor_vars], factor)
  
  # Construct the column name
  mrl_column_name <- paste0(frequency_band, "_mrl")
  
  # Assign the values of the desired column to mrl
  df$mrl <- df[[mrl_column_name]]
  
  #df <- df[df$period %in% c(0, 1), ]
  
  
  df <- df %>%
    group_by(animal, period_type, unit, neuron_type, period, group) %>%
    summarize(
      mrl = mean(mrl, na.rm = TRUE)
    ) %>%
    ungroup()

  if (average_unit) {
    averaged_over_unit <- df %>%
      group_by(animal, period_type, neuron_type, group) %>%
      summarize(
        mrl = mean(mrl, na.rm = TRUE)
      ) %>%
      ungroup()
    df <- averaged_over_unit
  }


  return(df) 
}

prepare_non_wavelet_df <- function(frequency_band, brain_region, average_unit=FALSE){
  
  csv_name = sprintf('mrl_%s_continuous_period_frequency_bins_%s.csv', frequency_band, brain_region)
  csv_file = paste(csv_dir, csv_name, sep='/')
  df <- read_csv(csv_file) 
  
  # Convert variables to factors
  factor_vars <- c('animal', 'group', 'period_type', 'neuron_type', 'unit')
  df[factor_vars] <- lapply(df[factor_vars], factor)
  
  # Construct the column name
  mrl_column_name <- paste0(frequency_band, "_mrl")
  
  # Assign the values of the desired column to mrl
  df$mrl <- df[[mrl_column_name]]

  
  if (average_unit) {
    averaged_over_unit <- df %>%
      group_by(animal, period_type, period, neuron_type, group) %>%
      summarize(
        mrl = mean(mrl, na.rm = TRUE)
      ) %>%
      ungroup()
    df <- averaged_over_unit
  }
  
  
  return(df) 
}


analyze_data <- function(frequency_band, brain_region, wavelet=FALSE, 
                         formula='mrl ~ group*neuron_type*period_type + (1|animal/unit)' ) 
  {
  if (wavelet) {
    data = prepare_wavelet_df(frequency_band, brain_region)
  } else {
    data = prepare_non_wavelet_df(frequency_band, brain_region)
  }
  model = lmer(formula=formula, data=data)
  plot = emmip(model, group ~ period_type | neuron_type, CIs = FALSE)
  plot = plot + scale_color_manual(values = c("green", "orange"))
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


summarize_data_by_means <- function(data) {
  
  data_means <- data  %>%
    group_by(group, neuron_type, period_type) %>%
    summarise(mean_mrl = mean(mean_mrl, na.rm = TRUE))
  
}


python_style_averaging <- function(data) {
# Step 1: Average over frequency_bin within levels of all other variables


avg_frequency_data <- data %>%
  group_by(group, animal, neuron_type, period_type) %>%
  summarise(mrl = mean(mrl, na.rm = TRUE))

final_avg_data <- avg_frequency_data %>%
  group_by(group, neuron_type, period_type) %>%
  summarise(mrl = mean(mrl, na.rm = TRUE))

return(final_avg_data)
}
group_colors = c("green", "orange")
make_model_plot <- function(model, group_colors) {
  plot = emmip(model, group ~ period_type | neuron_type, CIs = FALSE, col = group_colors)
  return(plot)
}

bar_plot <- function(data_means) {
  # Create the bar graph
  plot <- ggplot(data_means, aes(x = group, y = mrl, fill = group)) +
    geom_bar_pattern(aes(pattern = period_type),
                     stat = "identity", position = "dodge", width = 0.7,
                     pattern_density = 0.1,  # controls density of stripes
                     pattern_spacing = 0.02, # space between stripes
                     pattern_key_scale_factor = 1) +
    
    # Fill color for groups
    scale_fill_manual(values = c("control" = "green", "stressed" = "orange")) +
    
    # Facet by neuron_type
    facet_grid(neuron_type ~ ., scales = "free", space = "free") +
    
    # Theme and labels
    labs(title = "Group Averages by Condition", y = "Mean MRL") +
    theme_minimal()
  
  return(plot)
}




#### PL #####
### Delta ###

pl_delta_result = analyze_data('delta', 'pl')
summary(pl_delta_result$model)
pl_delta_result$plot

pl_delta_fx_result = analyze_data('delta', 'pl', formula='mrl ~ group*neuron_type*period_type + period +  (1|animal)')
summary(pl_delta_fx_result$model)
pl_delta_fx_result$plot




### Theta 1 ###

pl_theta_1_result = analyze_data('theta_1', 'pl')
summary(pl_theta_1_result$model)
pl_theta_1_result$plot

pl_theta_1_fx_result = analyze_data('theta_1', 'pl', formula='mrl ~ group*neuron_type*period_type + period +  (1|animal)')
summary(pl_theta_1_fx_result$model)
pl_theta_1_fx_result$plot


### Theta 2 ###

pl_theta_2_result = analyze_data('theta_2', 'pl')
summary(pl_theta_2_result$model)
pl_theta_2_result$plot

pl_theta_2_fx_result = analyze_data('theta_2', 'pl', formula='mrl ~ group*neuron_type*period_type + period +  (1|animal)')
summary(pl_theta_2_fx_result$model)
pl_theta_2_fx_result$plot


### Gamma ###

pl_gamma_result = analyze_data('gamma', 'pl')
summary(pl_gamma_result$model)
pl_gamma_result$plot

pl_gamma_fx_result = analyze_data('gamma', 'pl', formula='mrl ~ group*neuron_type*period_type + period +  (1|animal)')
summary(pl_gamma_fx_result$model)
pl_gamma_fx_result$plot


### HGamma ###

pl_hgamma_result = analyze_data('hgamma', 'pl')
summary(pl_hgamma_result$model)
pl_hgamma_result$plot

pl_hgamma_fx_result = analyze_data('hgamma', 'pl', formula='mrl ~ group*neuron_type*period_type + period +  (1|animal)')
summary(pl_hgamma_fx_result$model)
pl_hgamma_fx_result$plot






#### HPC #####

###Delta####

hpc_delta_result = analyze_data('delta', 'hpc')
summary(hpc_delta_result$model)
hpc_delta_result$plot

hpc_delta_fx_result = analyze_data('delta', 'hpc', formula='mrl ~ group*neuron_type*period_type + period +  (1|animal)')
summary(hpc_delta_fx_result$model) #trend effect here
hpc_delta_fx_result$plot

### Theta 1 ###

hpc_theta_1_result = analyze_data('theta_1', 'hpc')
summary(hpc_theta_1_result$model)
hpc_theta_1_result$plot

hpc_theta_1_fx_result = analyze_data('theta_1', 'hpc', formula='mrl ~ group*neuron_type*period_type + period +  (1|animal)')
summary(hpc_theta_1_fx_result$model)
hpc_theta_1_fx_result$plot


### Theta 2 ###

hpc_theta_2_result = analyze_data('theta_2', 'hpc')
summary(hpc_theta_2_result$model)
hpc_theta_2_result$plot

hpc_theta_2_fx_result = analyze_data('theta_2', 'hpc', formula='mrl ~ group*neuron_type*period_type + period + (1|animal)')
summary(hpc_theta_2_fx_result$model)
hpc_theta_2_fx_result$plot


### Gamma ###

hpc_gamma_result = analyze_data('gamma', 'hpc')
summary(hpc_gamma_result$model)
hpc_gamma_result$plot

hpc_gamma_fx_result = analyze_data('gamma', 'hpc', formula='mrl ~ group*neuron_type*period_type + period +  (1|animal)')
summary(hpc_gamma_fx_result$model)
hpc_gamma_fx_result$plot


### HGamma ###

hpc_hgamma_result = analyze_data('hgamma', 'hpc')
summary(hpc_hgamma_result$model)
hpc_hgamma_result$plot

hpc_hgamma_fx_result = analyze_data('hgamma', 'hpc', formula='mrl ~ group*neuron_type*period_type + period +  (1|animal)')
summary(hpc_hgamma_fx_result$model)
hpc_hgamma_fx_result$plot


#### BLA #####

#### Delta ####

bla_delta_result = analyze_data('delta', 'bla')
summary(hpc_delta_result$model)
hpc_delta_result$plot

bla_delta_fx_result = analyze_data('delta', 'bla', formula='mrl ~ group*neuron_type*period_type + period +  (1|animal)')
summary(bla_delta_fx_result$model)
bla_delta_fx_result$plot

### Theta 1 ###

bla_theta_1_result = analyze_data('theta_1', 'bla')
summary(bla_theta_1_result$model)
bla_theta_1_result$plot

bla_theta_1_fx_result = analyze_data('theta_1', 'bla', formula='mrl ~ group*neuron_type*period_type + period +  (1|animal)')
summary(bla_theta_1_fx_result$model)
bla_theta_1_fx_result$plot


### Theta 2 ###

bla_theta_2_result = analyze_data('theta_2', 'bla')
summary(bla_theta_2_result$model)
bla_theta_2_result$plot

bla_theta_2_fx_result = analyze_data('theta_2', 'bla', formula='mrl ~ group*neuron_type*period_type + period +  (1|animal)')
summary(bla_theta_2_fx_result$model)
bla_theta_2_fx_result$plot


### Gamma ###

bla_gamma_result = analyze_data('gamma', 'bla')
summary(bla_gamma_result$model)
bla_gamma_result$plot

bla_gamma_fx_result = analyze_data('gamma', 'bla', formula='mrl ~ group*neuron_type*period_type + period +  (1|animal)')
summary(bla_gamma_fx_result$model)
bla_gamma_fx_result$plot


### HGamma ###

bla_hgamma_result = analyze_data('hgamma', 'bla')
summary(bla_hgamma_result$model) # trend level
bla_hgamma_result$plot

bla_hgamma_fx_result = analyze_data('hgamma', 'bla', formula='mrl ~ group*neuron_type*period_type + period +  (1|animal)')
summary(bla_hgamma_fx_result$model) 
bla_hgamma_fx_result$plot



# Fit the Bayesian hierarchical model


