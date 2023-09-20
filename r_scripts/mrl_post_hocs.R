
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



quadrant_post_hocs <- function(data, brain_region, frequency_band) {
  
  # Define the variables and their levels
  variables <- list(
    neuron_type = c('IN', 'PN'),
    period_type = c('tone', 'pretone'),
    group = c('control', 'stressed')
  )
  
  # Inner function to fit the model and generate the result sentence
  generate_result <- function(subsetted_data, test_var, level1, level2) {
    formula <- as.formula(paste("mrl ~", test_var, "+ (1|animal/unit)"))
    model <- lmer(formula, data=subsetted_data)
    p_value <- summary(model)$coefficients[2,5]
    
    asterisks <- ifelse(p_value < 0.001, "***", ifelse(p_value < 0.01, "**", ifelse(p_value < 0.05, "*", "")))
    
    if (p_value < 0.05) {
      return(paste("Within ", brain_region, " ", frequency_band, " the effect of", test_var, "within", level1, "and", level2, "was significant at", p_value, asterisks, "."))
    } else {
      return(paste("Within ", brain_region, " ", frequency_band, " the effect of", test_var, "within", level1, "and", level2, "was not significant."))
    }
  }
  
  results <- list()
  
  # Loop through each combination of two variables
  for (var1 in names(variables)) {
    for (var2 in names(variables)) {
      if (var1 != var2) {
        test_var <- setdiff(names(variables), c(var1, var2))
        for (level1 in variables[[var1]]) {
          for (level2 in variables[[var2]]) {
            subsetted_data <- subset(data, (data[[var1]] == level1) & (data[[var2]] == level2))
            results[[paste(level1, level2)]] <- generate_result(subsetted_data, test_var, level1, level2)
          }
        }
      }
    }
  }
  
  return(results)
}

bla_theta_1_data <- prepare_non_wavelet_df('theta_1', 'bla')
bla_theta_1_quadrant_post_hoc_results <- quadrant_post_hocs(bla_theta_1_data, 'bla', 'theta_1')

print(quadrant_post_hoc_results)
 


# Call the function
results <- analyze_data(your_data)
print(results)



bla_theta_1_data <- prepare_non_wavelet_df('theta_1', 'bla')
bt1_control_PN <- subset(bla_theta_1_data , (group == 'control') & (neuron_type == 'PN'))
bt1_control_IN <- subset(bla_theta_1_data , (group == 'control') & (neuron_type == 'IN'))
bt1_stressed_PN <- subset(bla_theta_1_data , (group == 'stressed') & (neuron_type == 'PN'))
bt1_stressed_IN <- subset(bla_theta_1_data , (group == 'stressed') & (neuron_type == 'IN'))

bt1_posthoc_control_PN_model <- lmer(mrl ~ period_type + (1|animal/unit), data=bt1_control_PN) 
bt1_posthoc_control_IN_model <- lmer(mrl ~ period_type + (1|animal/unit), data=bt1_control_IN) 
bt1_posthoc_stressed_PN_model <- lmer(mrl ~ period_type + (1|animal/unit), data=bt1_stressed_PN) 
bt1_posthoc_stressed_IN_model <- lmer(mrl ~ period_type + (1|animal/unit), data=bt1_stressed_IN) 

summary(bt1_posthoc_control_PN_model)
summary(bt1_posthoc_control_IN_model)
summary(bt1_posthoc_stressed_PN_model)
summary(bt1_posthoc_stressed_IN_model)


bt1_tone_PN <- subset(bla_theta_1_data , (period_type == 'tone') & (neuron_type == 'PN'))
bt1_tone_IN <- subset(bla_theta_1_data , (period_type == 'tone') & (neuron_type == 'IN'))
bt1_pretone_PN <- subset(bla_theta_1_data , (period_type == 'pretone') & (neuron_type == 'PN'))
bt1_pretone_IN <- subset(bla_theta_1_data , (period_type == 'pretone') & (neuron_type == 'IN'))

bt1_posthoc_tone_PN_model <- lmer(mrl ~ group + (1|animal/unit), data=bt1_tone_PN) 
bt1_posthoc_tone_IN_model <- lmer(mrl ~ group + (1|animal/unit), data=bt1_tone_IN) 
bt1_posthoc_pretone_PN_model <- lmer(mrl ~ group + (1|animal/unit), data=bt1_pretone_PN) 
bt1_posthoc_pretone_IN_model <- lmer(mrl ~ group + (1|animal/unit), data=bt1_pretone_IN) 

summary(bt1_posthoc_tone_PN_model)
summary(bt1_posthoc_tone_IN_model)
summary(bt1_posthoc_pretone_PN_model)
summary(bt1_posthoc_pretone_IN_model)

bt1_tone <- subset(bla_theta_1_data , (period_type == 'tone') )
bt1_pretone <- subset(bla_theta_1_data , (period_type == 'pretone') )

bt1_posthoc_tone_model <- lmer(mrl ~ neuron_type + (1|animal/unit), data=bt1_tone)
bt1_posthoc_pretone_model <- lmer(mrl ~ neuron_type + (1|animal/unit), data=bt1_pretone)

summary(bt1_posthoc_tone_model)
summary(bt1_posthoc_pretone_model)

bt1_posthoc_tone_model_group <- lmer(mrl ~ group + (1|animal/unit), data=bt1_tone)
bt1_posthoc_pretone_model_group <- lmer(mrl ~ group + (1|animal/unit), data=bt1_pretone)


summary(bt1_posthoc_tone_model_group)
summary(bt1_posthoc_pretone_model_group)


bt1_IN <- subset(bla_theta_1_data , (neuron_type == 'IN') )
bt1_PN <- subset(bla_theta_1_data , (neuron_type == 'PN') )
bt1_posthoc_IN_model_group <- lmer(mrl ~ group + (1|animal/unit), data=bt1_IN)
bt1_posthoc_PN_model_group <- lmer(mrl ~ group + (1|animal/unit), data=bt1_PN)
summary(bt1_posthoc_IN_model_group)
summary(bt1_posthoc_PN_model_group)




bt1_posthoc_tone_x_model <- lmer(mrl ~ neuron_type * group + (1|animal/unit), data=bt1_tone)
bt1_posthoc_pretone_x_model <- lmer(mrl ~ neuron_type * group + (1|animal/unit), data=bt1_pretone)

summary(bt1_posthoc_tone_x_model)
summary(bt1_posthoc_pretone_x_model)

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

