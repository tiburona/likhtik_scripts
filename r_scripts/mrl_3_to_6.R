
prepare_non_wavelet_df_3_to_6 <- function(frequency_band, brain_region, average_unit=FALSE){
  
  csv_name = sprintf('mrl_%s_continuous_period_frequency_bins_-0.3_0.6_%s.csv', frequency_band, brain_region)
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
    data = prepare_non_wavelet_df_3_to_6(frequency_band, brain_region)
  }
  model = lmer(formula=formula, data=data)
  plot = emmip(model, group ~ period_type | neuron_type, CIs = FALSE)
  plot = plot + scale_color_manual(values = c("green", "orange"))
  return(list(model = model, plot = plot, data=data))
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