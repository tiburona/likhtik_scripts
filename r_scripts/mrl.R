library(glmmTMB)
library(lme4)
library(lmerTest)  # for lmer p-values
library(readr)
library(sjPlot)
library(ggplot2)
library(broom)
library(emmeans)
library(dplyr)

csv_dir = '/Users/katie/likhtik/data/lfp/mrl'

prepare_df <- function(csv){
  df <- read_csv(csv)
  
  # Convert variables to factors
  factor_vars <- c('animal', 'group', 'period_type', 'neuron_type', 'unit')
  df[factor_vars] <- lapply(df[factor_vars], factor)
  
  return(df)
}

analyze_data <- function(frequency_band, brain_region, wavelet=FALSE) {
  if (wavelet) {
    wavelet_string = 'wavelet_'
    error_term = '(1|animal/unit/period)'
  } else {
    wavelet_string = ''
    error_term = '(1|animal/unit)'
  }
  csv_name = sprintf('mrl_%s_continuous_period_frequency_bins_%s%s.csv', frequency_band, wavelet_string, brain_region)  # Assuming you want to append .csv to the frequency_band
  csv_file = paste(csv_dir, csv_name, sep='/')
  data = prepare_df(csv_file)
  formula = paste(frequency_band, '_mrl ~ group*neuron_type*period_type + ', error_term, sep='')
  # formula = paste('None_mrl ~ group*neuron_type*period_type + ', error_term, sep='')
  model = lmer(formula=formula, data=data)
  plot = emmip(model, group ~ period_type | neuron_type, CIs = FALSE)
  return(list(model = model, plot = plot))
}


#### PL #####
### Delta ###

pl_delta_result = analyze_data('delta', 'pl')
summary(pl_delta_result$model)
pl_delta_result$plot

pl_delta_wavelet_result = analyze_data('delta', 'pl', wavelet=TRUE)
summary(pl_delta_wavelet_result$model)
pl_delta_wavelet_result$plot


### Theta 1 ###

pl_theta_1_result = analyze_data('theta_1', 'pl')
summary(pl_theta_1_result$model)
pl_theta_1_result$plot

pl_theta_1_wavelet_result = analyze_data('theta_1', 'pl', wavelet=TRUE)
summary(pl_theta_1_wavelet_result$model)
pl_theta_1_wavelet_result$plot


### Theta 2 ###

pl_theta_2_result = analyze_data('theta_2', 'pl')
summary(pl_theta_2_result$model)
pl_theta_2_result$plot

pl_theta_2_wavelet_result = analyze_data('theta_2', 'pl', wavelet=TRUE)
summary(pl_theta_2_wavelet_result$model)
pl_theta_2_wavelet_result$plot


### Gamma ###

pl_gamma_result = analyze_data('gamma', 'pl')
summary(pl_gamma_result$model)
pl_gamma_result$plot

pl_gamma_wavelet_result = analyze_data('gamma', 'pl', wavelet=TRUE)
summary(pl_gamma_wavelet_result$model) # medium result here
pl_gamma_wavelet_result$plot


### HGamma ###

pl_hgamma_result = analyze_data('hgamma', 'pl')
summary(pl_hgamma_result$model)
pl_hgamma_result$plot

pl_hgamma_wavelet_result = analyze_data('hgamma', 'pl', wavelet=TRUE)
summary(pl_hgamma_wavelet_result$model) # medium result here
pl_hgamma_wavelet_result$plot



#### HPC #####
### Delta ###

hpc_delta_result = analyze_data('delta', 'hpc')
summary(hpc_delta_result$model) # very weak result here
hpc_delta_result$plot

hpc_delta_wavelet_result = analyze_data('delta', 'hpc', wavelet=TRUE)
summary(hpc_delta_wavelet_result$model)
hpc_delta_wavelet_result$plot


### Theta 1 ###

hpc_theta_1_result = analyze_data('theta_1', 'hpc')
summary(hpc_theta_1_result$model)
hpc_theta_1_result$plot

hpc_theta_1_wavelet_result = analyze_data('theta_1', 'hpc', wavelet=TRUE)
summary(hpc_theta_1_wavelet_result$model)
hpc_theta_1_wavelet_result$plot


### Theta 2 ###

hpc_theta_2_result = analyze_data('theta_2', 'hpc')
summary(hpc_theta_2_result$model) # weak result here
hpc_theta_2_result$plot

hpc_theta_2_wavelet_result = analyze_data('theta_2', 'hpc', wavelet=TRUE)
summary(hpc_theta_2_wavelet_result$model) # weak result here
hpc_theta_2_wavelet_result$plot


### Gamma ###

hpc_gamma_result = analyze_data('gamma', 'hpc')
summary(hpc_gamma_result$model)
hpc_gamma_result$plot

hpc_gamma_wavelet_result = analyze_data('gamma', 'hpc', wavelet=TRUE)
summary(hpc_gamma_wavelet_result$model) # strong result here
hpc_gamma_wavelet_result$plot


### HGamma ###

hpc_hgamma_result = analyze_data('hgamma', 'hpc')
summary(hpc_hgamma_result$model)
hpc_hgamma_result$plot

hpc_hgamma_wavelet_result = analyze_data('hgamma', 'hpc', wavelet=TRUE)
summary(hpc_hgamma_wavelet_result$model) # strong result here
hpc_hgamma_wavelet_result$plot


#### BLA #####
### Delta ###

bla_delta_result = analyze_data('delta', 'bla')
summary(bla_delta_result$model)
bla_delta_result$plot

bla_delta_wavelet_result = analyze_data('delta', 'bla', wavelet=TRUE)
summary(bla_delta_wavelet_result$model)
bla_delta_wavelet_result$plot


### Theta 1 ###

bla_theta_1_result = analyze_data('theta_1', 'bla')
summary(bla_theta_1_result$model)
bla_theta_1_result$plot

bla_theta_1_wavelet_result = analyze_data('theta_1', 'bla', wavelet=TRUE)
summary(bla_theta_1_wavelet_result$model)
bla_theta_1_wavelet_result$plot


### Theta 2 ###

bla_theta_2_result = analyze_data('theta_2', 'bla')
summary(bla_theta_2_result$model)
bla_theta_2_result$plot

bla_theta_2_wavelet_result = analyze_data('theta_2', 'bla', wavelet=TRUE)
summary(bla_theta_2_wavelet_result$model)
bla_theta_2_wavelet_result$plot

### Gamma ###

bla_gamma_result = analyze_data('gamma', 'bla')
summary(bla_gamma_result$model)
bla_gamma_result$plot

bla_gamma_wavelet_result = analyze_data('gamma', 'bla', wavelet=TRUE)
summary(bla_gamma_wavelet_result$model) # strong result here
bla_gamma_wavelet_result$plot


### HGamma ###

bla_hgamma_result = analyze_data('hgamma', 'bla')
summary(bla_hgamma_result$model)
bla_hgamma_result$plot

bla_hgamma_wavelet_result = analyze_data('hgamma', 'bla', wavelet=TRUE)
summary(bla_hgamma_wavelet_result$model) # strong result here
bla_hgamma_wavelet_result$plot

print_duplicates <- function(data, column_name) {
  # Identify rows with duplicate values in the specified column
  duplicated_rows <- data[duplicated(data[[column_name]]) | 
                            duplicated(data[[column_name]], fromLast = TRUE), ]
  
  # If there are any duplicates, print the other columns' names and values
  if(nrow(duplicated_rows) > 0) {
    for(i in 1:nrow(duplicated_rows)) {
      cat("Row with duplicate value in", column_name, "\n")
      for(col in names(duplicated_rows)) {
        if(col != column_name) {
          value <- duplicated_rows[i, col]
          if(is.list(value) || is.data.frame(value)) {
            value <- paste(capture.output(print(value)), collapse = " ")
          }
          cat(col, ":", value, "\n")
        }
      }
      cat("\n")
    }
  } else {
    cat("No duplicates found in the column", column_name, "\n")
  }
}
# 
# # Call the function
# column_name <- "hgamma_mrl"  
# print_duplicates(bla_hgamma_df, column_name)
# 
# # Call the function for your column name
# bla_hgamma_df = prepare_df('/Users/katie/likhtik/data/lfp/mrl/mrl_hgamma_continuous_period_frequency_bins_bla.csv')
# column_name <- "hgamma_mrl"  # Replace with the name of your column
# print_duplicates(bla_hgamma_df, column_name)