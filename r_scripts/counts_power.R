library(lme4)
library(nlme)
library(dplyr)
library(glmmTMB)
library(lmerTest)
library(sjPlot)
library(ggplot2)
library(emmeans)
library(DHARMa)
library(boot)



# Function to plot aggregated actual data
plot_aggregated_data <- function(data, x, y, xlabel, ylabel, factors=c("neuron_type", "period_type", "group")) {
  # Create bins for the continuous IV based on quantiles
  
  data <- subset(data, !is.na(data[[x]]))
  
  data <- data %>%
    mutate(bin = ntile(get(x), 10))  # Creating 10 bins
  
  if ("period_type" %in% factors && "period_type" %in% names(data)) {
    data$period_type <- factor(data$period_type, levels = c("tone", "pretone"))
  }
  
 
  
  # Dynamically construct the grouping variables
  grouping_vars <- c("bin", "animal", "unit", "period", factors)
  
  # Compute averages at each level
  data <- data %>%
    group_by(across(all_of(grouping_vars))) %>%
    summarise(mean_y_period = median(get(y), na.rm = TRUE), .groups = 'drop') %>%
    group_by(across(setdiff(grouping_vars, "period"))) %>%
    summarise(mean_y_unit = median(mean_y_period, na.rm = TRUE), .groups = 'drop') %>%
    group_by(across(setdiff(grouping_vars, c("period", "unit")))) %>%
    summarise(mean_y_animal = median(mean_y_unit, na.rm = TRUE), .groups = 'drop') %>%
    group_by(across(setdiff(grouping_vars, c("period", "unit", "animal")))) %>%
    summarise(mean_y_group = median(mean_y_animal, na.rm = TRUE), .groups = 'drop')
  
  # Add a new interaction column for plotting
  data <- mutate(data, interaction_column = interaction(!!!syms(factors)))
  
  data <- mutate(data, color_for_period_type = gsub(".*\\b(pretone|tone)\\b.*", "\\1", interaction_column))
  
  
  
  # Setup plot aesthetics based on factors
  color_var <- if("period_type" %in% factors) "period_type" else "group"
  colors <- if(color_var == "period_type") {
    c("tone" = "green", "pretone" = "pink")
  } else {
    c("control" = "purple", "defeat" = "orange")
  }
  
  # Setup faceting based on factors
  facet_formula <- switch(length(factors),
                          '1' = paste0(". ~ ", factors),
                          '2' = paste0(factors[1], " ~ ", factors[2]),
                          '3' = "neuron_type ~ group",
                          '. ~ group')  # Default fall-back to free scale on group
  
  # Create the plot using the correct interaction column
  color_var <- if("period_type" %in% factors) "period_type" else "group"
  # Create the plot
  p <- ggplot(data, aes(x = as.factor(bin), y = mean_y_group, color = color_for_period_type, group = interaction_column)) +
    geom_line() +
    labs(x = xlabel, y = ylabel) +
    scale_color_manual(values = colors) +
    theme_bw() +
    facet_grid(facet_formula, scales = "free")
  
  return(p)
  
  
}



create_predictions_data_no_nt <- function(data, model, continuous_predictor) {
  
  
  mean_iv <- mean(data[[continuous_predictor]], na.rm = TRUE)
  sd_iv <- sd(data[[continuous_predictor]], na.rm = TRUE)
  
  pred_data <- expand.grid(
    group = levels(clean_data$group),
    period_type = levels(clean_data$period_type),
    iv = c(mean_iv - sd_iv, mean_iv, mean_iv + sd_iv)
  )
  
  # Properly name the power variable in the prediction data
  names(pred_data)[names(pred_data) == "iv"] <- continuous_predictor
  
  pred_data$predicted <- predict(model, newdata = pred_data, re.form = NA)
  
  return(pred_data)
}


run_count_power_analysis <- function(data, region, frequency_band) {
  # Dynamically build the variable names based on region and frequency
  power_var <- paste(region, frequency_band, "power", sep = "_")
  
  # Filter out rows where the power variable is NA
  clean_data <- data[!is.na(data[[power_var]]), ]
  
  # Build the formula string dynamically
  formula <- as.formula(
    paste("count ~ group*period_type*neuron_type*", power_var,
    "+ (1|animal:unit) + (1|animal:unit:period)", sep = ""))

  model <- glmmTMB(formula, ziformula = ~ 1,family = nbinom1, data = clean_data)
  
  disp_stat <- sum(residuals(model, type = "pearson")^2) / df.residual(model)
  print(paste("Dispersion statistic: ", disp_stat))
  
  plot(residuals(model) ~ fitted(model))
  abline(h = 0, col = "red")

  # Display the summary of the model
  print(summary(model))
  
  # Calculate mean and standard deviation for the power variable
  mean_power <- mean(clean_data[[power_var]], na.rm = TRUE)
  sd_power <- sd(clean_data[[power_var]], na.rm = TRUE)
  
  # First, create a unique identifier in your training data for existing combinations
  clean_data$combo <- with(clean_data, paste(animal, unit, period, sep = "_"))
  
  # Generate prediction data grid
  pred_data <- expand.grid(
    group = levels(clean_data$group),
    period_type = levels(clean_data$period_type),
    neuron_type = levels(clean_data$neuron_type),
    power = c(mean_power - sd_power, mean_power, mean_power + sd_power),
    animal = unique(clean_data$animal),
    unit = unique(clean_data$unit),
    period = unique(clean_data$period)
  )
  
  # Create a combo identifier in the prediction data
  pred_data$combo <- with(pred_data, paste(animal, unit, period, sep = "_"))
  
  # Filter pred_data to only include combinations that exist in clean_data
  pred_data <- pred_data[pred_data$combo %in% clean_data$combo,]
  
  # Properly name the power variable in the prediction data
  names(pred_data)[names(pred_data) == "power"] <- power_var
  
  # Proceed with predictions
  pred_data$predicted_counts <- predict(model, newdata = pred_data, re.form = NA,
                                        type = 'response')
  
  # Plot the predictions
  p <- graph_predictions(
    data=pred_data, x=power_var, y='predicted_counts', 
    xlabel=paste(toupper(region), frequency_band, "power", sep = " "),
    ylabel= labs(y = "Predicted Count of Spikes per Event (0-.3s)")) 

  # Return both the model and the plot
  return(list(model = model, plot = p))
}




### .3 second periods analyses ###

csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/spike_counts', 'count_power.csv', sep='/')
data <- read.csv(csv, comment.char="#") 

read_metadata(csv)



factor_vars <- c('animal', 'group', 'period_type', 'neuron_type', 'unit', 'period')
data[factor_vars] <- lapply(data[factor_vars], factor)

data$neuron_type <- factor(data$neuron_type,
                           levels = c("IN", "PN"))

na_data <- data[is.na(data$neuron_type), ]
data <- data[!is.na(data$neuron_type), ]


data$hpc_theta_1_power <- scale(data$hpc_theta_1_power)
data$bla_theta_1_power <- scale(data$bla_theta_1_power)
data$pl_theta_1_power  <- scale(data$pl_theta_1_power)
data$hpc_theta_2_power <- scale(data$hpc_theta_2_power)
data$bla_theta_2_power <- scale(data$bla_theta_2_power)
data$pl_theta_2_power  <- scale(data$pl_theta_2_power)


data_with_counts <- data %>%
  filter(time < .3) %>%
  group_by(period, group, period_type, neuron_type, unit, animal, event) %>%
  summarise(
    hpc_theta_1_power = mean(hpc_theta_1_power, na.rm = TRUE),
    bla_theta_1_power = mean(bla_theta_1_power, na.rm = TRUE),
    pl_theta_1_power  = mean(pl_theta_1_power, na.rm = TRUE),
    hpc_theta_2_power = mean(hpc_theta_2_power, na.rm = TRUE),
    bla_theta_2_power = mean(bla_theta_2_power, na.rm = TRUE),
    pl_theta_2_power  = mean(pl_theta_2_power, na.rm = TRUE),
    count = sum(spike_counts, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )


### BLA THETA 1 ###

# clean_data <- data_with_counts[!is.na(data_with_counts[['bla_theta_1_power']]), ]
# results <- run_count_power_analysis(data = clean_data, region = "bla", frequency_band = "theta_1")
# print(results$plot)
# 
# 
# ### BLA THETA 2 ###
# 
# clean_data <- data_with_counts[!is.na(data_with_counts[['bla_theta_2_power']]), ]
# results <- run_count_power_analysis(data = clean_data, region = "bla", frequency_band = "theta_2")
# print(results$plot)


### PL THETA 1 ###

clean_data <- data_with_counts[!is.na(data_with_counts[['pl_theta_1_power']]), ]
results <- run_count_power_analysis(data = clean_data, region = "pl", frequency_band = "theta_1")
print(results$plot)

plot_aggregated_data(clean_data, 'pl_theta_1_power', 'count', "PL Theta 1 Power", "Count", factors=c("neuron_type", "period_type", "group"))

continuous_data <- data %>%
  filter(time < .3) %>%
  filter(!is.na(pl_theta_1_power)) %>%
  group_by(period, group, period_type, neuron_type, unit, animal, event, time_bin) %>%
  summarise(
    pl_theta_1_power  = mean(pl_theta_1_power, na.rm = TRUE),
    count = sum(spike_counts, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )

continuous_data <- continuous_data %>%
  arrange(animal, unit, period, group, period_type, neuron_type, event, time_bin) %>%
  group_by(animal, unit, period, group, period_type, neuron_type, event) %>%
  mutate(count_lag1 = lag(count, default = NA)) %>%
  ungroup()


continuous_data <- continuous_data %>%
  group_by(animal, unit, period, group, period_type, neuron_type, event) %>%
  filter(time_bin != min(time_bin)) %>%
  ungroup()

continuous_data <- continuous_data %>%
  mutate(across(where(is.factor), droplevels))





model <- glmmTMB(count ~ time_bin + count_lag1 + group * period_type * neuron_type * pl_theta_1_power + 
                   (1 | animal:unit) + (1 | animal:unit:period:event),
                 data = continuous_data, 
                 family = poisson(link = "log"),
                 ziformula = ~0,  # No zero-inflation
                 na.action = na.exclude,
                 control = glmmTMBControl(optimizer = optim, optArgs = list(method = "BFGS")))

resid <- residuals(model, type = "response")
plot(resid, main = "Residuals Plot")

fitted_values <- fitted(model)
plot(fitted_values, resid, xlab = "Fitted values", ylab = "Residuals", main = "Residuals vs. Fitted Values")
abline(h = 0, col = "red")

observed_deviance <- sum(residuals(model, type = "pearson")^2)
df_residual <- df.residual(model)
overdispersion_factor = observed_deviance / df_residual
print(paste("Overdispersion factor:", overdispersion_factor))


# Simulate residuals using DHARMa
simulationOutput <- simulateResiduals(fittedModel = model)

plot(simulationOutput, quantreg = TRUE) 

# Alternatively, manually creating a Q-Q plot
# Predicted counts
fitted_counts <- fitted(model)

# Sort both observed counts and fitted counts
observed_sorted <- sort(model$data$count)
theoretical_quantiles <- qpois(ppoints(length(fitted_counts)), lambda = fitted_counts)

# Q-Q plot
plot(theoretical_quantiles, observed_sorted, main = "Q-Q Plot for Poisson Distribution",
     xlab = "Theoretical Quantiles", ylab = "Observed Quantiles")
abline(0, 1, col = "red")  # Line y = x for reference





### PL THETA 2 ###

clean_data <- data_with_counts[!is.na(data_with_counts[['pl_theta_2_power']]), ]
results <- run_count_power_analysis(data = clean_data, region = "pl", frequency_band = "theta_2")
print(results$plot)


### HPC THETA 1 ###

clean_data <- data_with_counts[!is.na(data_with_counts[['hpc_theta_1_power']]), ]
results <- run_count_power_analysis(data = clean_data, region = "hpc", frequency_band = "theta_1")
print(results$plot)


### HPC THETA 2 ###

clean_data <- data_with_counts[!is.na(data_with_counts[['hpc_theta_2_power']]), ]
results <- run_count_power_analysis(data = clean_data, region = "hpc", frequency_band = "theta_2")
print(results$plot)


