
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


post_hocs <- function(data, brain_region, frequency_band, division = "quadrants") {
  
  # Define the variables and their levels
  variables <- list(
    neuron_type = c('IN', 'PN'),
    period_type = c('tone', 'pretone'),
    group = c('control', 'stressed')
  )
  
  # Inner function to fit the model and generate the result sentence
  generate_result <- function(subsetted_data, fixed_effect, level1, level2 = NULL) {
    formula <- as.formula(paste("mrl ~", fixed_effect, "+ (1|animal/unit)"))
    model <- lmer(formula, data=subsetted_data)
    p_value <- summary(model)$coefficients[2,5]
    
    asterisks <- ifelse(p_value < 0.001, "***", ifelse(p_value < 0.01, "**", ifelse(p_value < 0.05, "*", "")))
    
    if (is.null(level2)) {
      sentence <- paste("In", brain_region, frequency_band, "the effect of", fixed_effect, "within", level1)
    } else {
      sentence <- paste("In", brain_region, frequency_band, "the effect of", fixed_effect, "within", level1, "and", level2)
    }
    
    if (p_value < 0.05) {
      return(paste(sentence, "was significant at", p_value, asterisks, "."))
    } else {
      return(paste(sentence, "was not significant."))
    }
  }
  
  results <- list()
  
  if (division == "quadrants") {
    for (var1 in names(variables)) {
      for (var2 in names(variables)) {
        if (var1 != var2) {
          test_var <- setdiff(names(variables), c(var1, var2))
          for (level1 in variables[[var1]]) {
            for (level2 in variables[[var2]]) {
              subsetted_data <- subset(data, (data[[var1]] == level1) & (data[[var2]] == level2))
              results[[paste(level1, level2, test_var)]] <- generate_result(subsetted_data, test_var, level1, level2)
            }
          }
        }
      }
    }
  } else if (division == "halves") {
    for (var in names(variables)) {
      for (level in variables[[var]]) {
        subsetted_data <- subset(data, data[[var]] == level)
        remaining_vars <- setdiff(names(variables), var)
        for (fixed_effect in remaining_vars) {
          results[[paste(level, fixed_effect)]] <- generate_result(subsetted_data, fixed_effect, level)
        }
      }
    }
  }
  
  return(results)
}

bla_theta_1_data <- prepare_non_wavelet_df('theta_1', 'bla')
bla_theta_1_quadrant_post_hoc_results <- post_hocs(bla_theta_1_data, 'bla', 'theta_1')


print(bla_theta_1_quadrant_post_hoc_results)

bla_theta_1_halves_post_hoc_results <- post_hocs(bla_theta_1_data, 'bla', 'theta_1', division='halves')
print(bla_theta_1_halves_post_hoc_results)

 


hpc_theta_1_data <- prepare_non_wavelet_df('theta_1', 'hpc')
hpc_theta_1_quadrant_post_hoc_results <- post_hocs(hpc_theta_1_data, 'hpc', 'theta_1')


print(hpc_theta_1_quadrant_post_hoc_results)

hpc_theta_1_halves_post_hoc_results <- post_hocs(hpc_theta_1_data, 'hpc', 'theta_1', division='halves')
print(hpc_theta_1_halves_post_hoc_results)

print(hpc_theta_1_halves_post_hoc_results)



pl_theta_1_data <- prepare_non_wavelet_df('theta_1', 'pl')
pl_theta_1_quadrant_post_hoc_results <- post_hocs(pl_theta_1_data, 'pl', 'theta_1')


print(pl_theta_1_quadrant_post_hoc_results)

pl_theta_1_halves_post_hoc_results <- post_hocs(hpc_theta_1_data, 'pl', 'theta_1', division='halves')
print(pl_theta_1_halves_post_hoc_results)

print(pl_theta_1_halves_post_hoc_results)






