library(lme4)
library(nlme)
library(dplyr)
library(glmmTMB)
library(lmerTest)
library(sjPlot)
library(ggplot2)

#### HIGH THETA POWER RELATIONSHIP WITH FIRING RATE ###

average_data <- function(data, brain_region, deviation_standard=">=2", group_over_period=FALSE) {
  
  # Check if brain_region is a character string
  if (!is.character(brain_region) || length(brain_region) != 1) {
    stop("brain_region must be a single character string")
  }
  
  # Create the deviation indicator variable name
  deviation_indicator_var <- paste0(brain_region, "_theta_1_power_deviation_indicator")
  
  factor_vars <- c('animal', 'group', 'period_type', 'neuron_type', 'unit')
  data[factor_vars] <- lapply(data[factor_vars], factor)
  
  # Check if deviation_indicator_var exists in the data
  if (!(deviation_indicator_var %in% names(data))) {
    stop(paste0(deviation_indicator_var, " not found in the data"))
  }
  
  # Step 1: Take a subset of data where an indicator variable is ">=2" and time_bin <30
  subset_data <- data %>%
    filter((.data[[deviation_indicator_var]] == deviation_standard) & time_bin < 30)
  
  # Step 2: Identify columns with 'power' in their name but not 'indicator'
  power_vars <- grep("power", names(subset_data), value = TRUE)
  power_vars <- power_vars[!grepl("indicator", power_vars)]
  
  # Step 3: Average over time bins and group by multiple variables
  # Make sure to carry over variables that do not change within a trial
  # Set base grouping variables
  grouping_vars <- c("group", "animal", "period", "unit", "period_type", "neuron_type")
  
  # Conditionally add 'trial' if group_over_period is FALSE
  if (!group_over_period) {
    grouping_vars <- c("trial", grouping_vars)
  }
  
  averaged_data <- subset_data %>%
    group_by(across(all_of(grouping_vars))) %>%
    summarise(
      across(all_of(power_vars), mean, .names = "mean_{col}", na.rm = TRUE),
      rate_mean = mean(rate, na.rm = TRUE),
      .groups = "drop"
    )
  
  return(averaged_data)
}


deviation_csv = paste('/Users/katie/likhtik/data/lfp/percent_freezing', 'theta_power_deviations.csv', sep='/')
deviation_data <- read.csv(deviation_csv, comment.char="#") 


# Get column names
col_names <- colnames(deviation_data)

# Iterate through each column name
for (col_name in col_names) {
  # Check if "deviation" is in the column name
  if (grepl("deviation", col_name)) {
    # Create a new column name for the indicator variable
    new_col_name <- paste0(col_name, "_indicator")
    
    # Create a new column based on the condition
    deviation_data[[new_col_name]] <- ifelse(deviation_data[[col_name]] >= 2, ">=2",
                                             ifelse(deviation_data[[col_name]] <= -2, "<=-2", "in between"))
  }
}



analyze_and_plot <- function(averaged_data, brain_region) {
  
  # Check if brain_region is a character string
  if (!is.character(brain_region) || length(brain_region) != 1) {
    stop("brain_region must be a single character string")
  }
  
  # Create the variable names dynamically
  power_var <- paste0("mean_", brain_region, "_theta_1_power")
  
  # Check if power_var exists in the data
  if (!(power_var %in% names(averaged_data))) {
    stop(paste0(power_var, " not found in the data"))
  }
  
  # Fit the model
  model_formula <- as.formula(paste("rate_mean ~", power_var, "* period_type * neuron_type * group + (1|animal/unit)"))
  model <- lmer(model_formula, data=averaged_data)
  print(summary(model))
  
  # Create new data for prediction
  new_data <- expand.grid(
    mean_power = c(mean(averaged_data[[power_var]], na.rm=TRUE) - sd(averaged_data[[power_var]], na.rm=TRUE), 
                   mean(averaged_data[[power_var]], na.rm=TRUE), 
                   mean(averaged_data[[power_var]], na.rm=TRUE) + sd(averaged_data[[power_var]], na.rm=TRUE)),
    group = unique(averaged_data$group),
    period_type = unique(averaged_data$period_type),
    neuron_type = unique(averaged_data$neuron_type)
  )
  
  # Rename for prediction
  names(new_data)[names(new_data) == "mean_power"] <- power_var
  
  # Predict
  new_data$predicted_rate <- predict(model, newdata = new_data, re.form = NA)
  
  # Rename back for plotting
  names(new_data)[names(new_data) == power_var] <- "mean_power"
  
  # Plot
  p <- ggplot(new_data, aes(x = mean_power, y = predicted_rate, color = period_type)) +
    geom_line() +
    labs(x = paste(toupper(substr(brain_region, 1, 1)), substr(brain_region, 2, nchar(brain_region)), " Theta 1 Power", sep = ""),
         y = "Predicted Firing Rate") +
    theme_bw() +
    facet_grid(neuron_type ~ group, scales = "free")
  
  # Return the plot and model
  return(list(plot = p, model = model))
}




###BLA###

avg_bla_data <- average_data(deviation_data, 'bla')
bla_results <- analyze_and_plot(avg_bla_data, 'bla')

# To display the plot
print(bla_results$plot)



###PL###
avg_pl_data <- average_data(deviation_data, 'pl')
pl_results <- analyze_and_plot(avg_pl_data, 'pl')

# To display the plt
print(pl_results$plot)

###HPC###
avg_hpc_data <- average_data(deviation_data, 'hpc')
hpc_results <- analyze_and_plot(avg_hpc_data, 'hpc')

# To display the plt
print(hpc_results$plot)


### AVERAGE SPIKE DATA FOR HIGH AND LOW POWER ###

bla_hi_average_data = average_data(deviation_data, 'bla', deviation_standard=">=2", group_over_period=TRUE)



### JUST POWER GRAPHS ###



mean_data <- deviation_data %>%
  filter(time_bin < 30) %>%
  group_by(period, group, period_type) %>%
  summarise(
    mean_hpc_theta_1_power = mean(hpc_theta_1_power, na.rm = TRUE),
    mean_bla_theta_1_power = mean(bla_theta_1_power, na.rm = TRUE),
    mean_pl_theta_1_power  = mean(pl_theta_1_power, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )


# BLA #

ggplot(mean_data, aes(x = period, y = mean_bla_theta_1_power, color = period_type, shape = period_type)) +
  geom_point(aes(shape = period_type, color = period_type), size = 3) +
  geom_line(aes(group = period_type)) +
  scale_shape_manual(values = c("pretone" = 15, "tone" = 16)) + # 15 is for squares and 16 is for circles
  scale_color_manual(values = c("pretone" = "gray", "tone" = "blue")) +
  labs(x = "Period", y = "Mean BLA Theta 1 Power") +
  facet_wrap(~ group, ncol = 1) + # Separate panels for each level of 'group'
  theme_minimal()

# PL #

ggplot(mean_data, aes(x = period, y = mean_pl_theta_1_power, color = period_type, shape = period_type)) +
  geom_point(aes(shape = period_type, color = period_type), size = 3) +
  geom_line(aes(group = period_type)) +
  scale_shape_manual(values = c("pretone" = 15, "tone" = 16)) + # 15 is for squares and 16 is for circles
  scale_color_manual(values = c("pretone" = "gray", "tone" = "blue")) +
  labs(x = "Period", y = "Mean PL Theta 1 Power") +
  facet_wrap(~ group, ncol = 1) + # Separate panels for each level of 'group'
  theme_minimal()


