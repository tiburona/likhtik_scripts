library(lme4)
library(dplyr)
library(ggplot2)
library(emmeans)
library(glmmTMB)
library(DHARMa)
library(groupdata2)


csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/percent_freezing', 'counts_freezing.csv', sep='/')


data <- read.csv(csv, comment.char="#") 


factor_vars <- c('animal', 'group', 'period_type', 'unit', 'neuron_type')
data[factor_vars] <- lapply(data[factor_vars], factor)


# Filter data for pretone and tone period types
data <- subset(data, period_type %in% c("pretone", "tone"))

data <- data[!is.na(data$percent_freezing), ]


data$neuron_type <- factor(data$neuron_type,
                           levels = c("IN", "PN"))

na_data <- data[is.na(data$neuron_type), ]
data <- data[!is.na(data$neuron_type), ]





freezing_counts_data <- data %>%
  filter(time_bin < 30) %>%
  group_by(period, group, period_type, neuron_type, unit, animal, event) %>%
  summarise(
    count = sum(spike_counts, na.rm = TRUE),
    percent_freezing = mean(percent_freezing, na.rm = TRUE),
    .groups = "drop"
  )



median_val <- median(freezing_counts_data$percent_freezing)
iqr_val <- IQR(freezing_counts_data$percent_freezing)

freezing_counts_data$scaled_percent_freezing <- (freezing_counts_data$percent_freezing - median_val) / iqr_val


# Plotting histogram of original percent freezing data
ggplot(freezing_counts_data, aes(x = percent_freezing)) +
  geom_histogram(bins = 30, fill = "blue", color = "black") +
  labs(title = "Histogram of Percent Freezing")



ggplot(freezing_counts_data, aes(x = scaled_percent_freezing, y = count)) +
  geom_point(alpha = 0.6) +  # Adjust alpha for point transparency if data points overlap
  labs(x = "Percent Freezing", y = "Firing Count",
       title = "Scatterplot of Percent Freezing vs. Firing Count") +
  theme_minimal()  # Using a minimal theme for a clean look


# # Function to plot aggregated actual data
# plot_aggregated_data <- function(data, x, y, xlabel, ylabel, num_vars=4) {
#   # Create bins for the continuous IV based on quantiles
#   data <- data %>%
#     mutate(bin = ntile(get(x), 10))  # Creating 5 bins
#   
#   # Compute average y within each period first
#   data <- data %>%
#     group_by(bin, group, neuron_type, period_type, animal, unit, period) %>%
#     summarise(mean_y_period = mean(get(y), na.rm = TRUE), .groups = 'drop')
#   
#   # Compute average of period averages within each unit
#   data <- data %>%
#     group_by(bin, group, neuron_type, period_type, animal, unit) %>%
#     summarise(mean_y_unit = mean(mean_y_period, na.rm = TRUE), .groups = 'drop')
#   
#   # Compute average of unit averages within each animal
#   data <- data %>%
#     group_by(bin, group, neuron_type, period_type, animal) %>%
#     summarise(mean_y_animal = mean(mean_y_unit, na.rm = TRUE), .groups = 'drop')
#   
#   # Compute average of animal averages within each group
#   data <- data %>%
#     group_by(bin, group, neuron_type, period_type) %>%
#     summarise(mean_y_group = mean(mean_y_animal, na.rm = TRUE), .groups = 'drop')
#   
#   # Graph the aggregated data
#   p <- ggplot(data, aes(x = as.factor(bin), y = mean_y_group, color = period_type, group = interaction(group, neuron_type, period_type))) +
#     geom_line() +
#     labs(x = xlabel, y = ylabel) +
#     scale_color_manual(values = c("tone" = "green", "pretone" = "pink")) +
#     theme_bw()
#   
#   # Adjust facets based on the number of variables
#   if (num_vars == 4) {
#     p <- p + facet_grid("neuron_type ~ group", scales = "free")
#   } else {
#     p <- p + facet_grid(". ~ group", scales = "free")
#   }
#   
#   return(p)
# }





# Function to plot aggregated actual data
plot_aggregated_data <- function(data, x, y, xlabel, ylabel, factors=c("neuron_type", "period_type", "group")) {
  # Create bins for the continuous IV based on quantiles
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
  
  data <- mutate(data, color_for_period_type = gsub("([^.]*).*", "\\1", interaction_column))
  
  
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

plot_aggregated_data(freezing_counts_data, "scaled_percent_freezing", "count", "Scaled Percent Freezing", "Mean Count", factors=c("period_type", "group", "neuron_type"))

# Calculate mean and standard deviation for the power variable
mean_freezing <- mean(freezing_counts_data$scaled_percent_freezing, na.rm = TRUE)
sd_freezing <- sd(freezing_counts_data$scaled_percent_freezing, na.rm = TRUE)

run_count_freezing_analysis <- function(data, formula) {
  
  clean_data <- data[!is.na(data$scaled_percent_freezing), ]
  
  clean_data$combo <- with(clean_data, paste(animal, unit, period, sep = "_"))
  
  
  model <- glmmTMB(as.formula(formula), family = nbinom1, data = clean_data)
  
  print(summary(model))
  
  
  # Generate prediction data grid
  pred_data <- expand.grid(
    group = levels(clean_data$group),
    period_type = levels(clean_data$period_type),
    neuron_type = levels(clean_data$neuron_type),
    scaled_percent_freezing = c(mean_freezing - sd_freezing, mean_freezing, mean_freezing + sd_freezing),
    animal = unique(clean_data$animal),
    unit = unique(clean_data$unit),
    period = unique(clean_data$period)
  )
  
  # Create a combo identifier in the prediction data
  pred_data$combo <- with(pred_data, paste(animal, unit, period, sep = "_"))
  
  # Filter pred_data to only include combinations that exist in clean_data
  pred_data <- pred_data[pred_data$combo %in% clean_data$combo,]
  
  # Proceed with predictions
  pred_data$predicted_counts <- predict(model, newdata = pred_data, re.form = NA,
                                        type = 'response')
  
  # Plot the predictions
  p <- graph_predictions(
    data=pred_data, x='scaled_percent_freezing', y='predicted_counts', 
    xlabel="Scaled Percent Freezing",
    ylabel= labs(y = "Predicted Count of Spikes per Event (0-.3s)")) 
  
  # Return both the model and the plot
  return(list(model = model, plot = p))
  
}

formula_start <- 'count ~ group * period_type * neuron_type * scaled_percent_freezing'

model_formula_random_slope <- paste(formula_start, ' + (1|animal:unit:period) + (1 + period_type|animal:unit)')

model_formula_no_random_slope <- paste(formula_start, ' + (1|animal:unit:period) + (1|animal:unit)')

random_slope_results <- run_count_freezing_analysis(freezing_counts_data, 
                                                    model_formula_random_slope)

no_random_slope_results <- run_count_freezing_analysis(freezing_counts_data, 
                                                       model_formula_no_random_slope)


summary(random_slope_results$model)

random_slope_results$plot
no_random_slope_results$plot

simulationOutput <- simulateResiduals(fittedModel = model)
plotSimulatedResiduals(simulationOutput, quantreg = TRUE)





# Calculate model residuals
residuals <- residuals(model, type = "pearson")

# Estimate dispersion parameter
dispersion_estimate <- sum(residuals^2) / df.residual(model)

# Output the dispersion estimate
print(dispersion_estimate)
# A value close to 1 suggests appropriate dispersion handling.





# Checking residuals for patterns or anomalies
residuals <- residuals(model, type = "pearson")

# Plotting residuals against fitted values
plot(fitted(model), residuals, xlab = "Fitted Values", ylab = "Pearson Residuals")
abline(h = 0, col = "red")

# Adding a title for clarity
title("Residuals vs Fitted Values Plot")



residual_data <- data.frame(
  Fitted = fitted(model),
  Residuals = residuals(model, type = "pearson")
)

freezing_counts_data$RowID <- seq_along(freezing_counts_data[[1]])  # Assuming the first column can be used to count rows

combined_data <- cbind(freezing_counts_data, residual_data)


ggplot(combined_data, aes(x = Fitted, y = Residuals)) +
  geom_point(aes(color = factor(combined_data$group), size = combined_data$scaled_percent_freezing)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals vs. Fitted Values",
       x = "Fitted Values",
       y = "Residuals",
       color = "Period Type",
       size = "Scaled Percent Freezing") +
  theme_minimal()


significant_points <- combined_data[abs(combined_data$Residuals) > 5, ]

# Adding labels to the plot
ggplot(combined_data, aes(x = Fitted, y = Residuals)) +
  geom_point(aes(color = factor(combined_data$neuron_type), size = combined_data$scaled_percent_freezing)) +
  geom_text(data = significant_points, aes(label = animal), hjust = 1.5, vjust = 1.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "blue") +
  labs(title = "Identifying Outliers in Residuals",
       x = "Fitted Values",
       y = "Residuals")



fit_and_evaluate_model <- function(train_data, test_data) {
  model <- glmmTMB(count ~ group * period_type * neuron_type * scaled_percent_freezing + 
                     (1|animal:unit:period) + (1 + period_type|animal:unit),
                   family = nbinom1, data = train_data)
  
  # Predicting on test data
  pred <- predict(model, newdata = test_data, type = "response")
  
  # Calculate correlation between predicted and actual counts
  actual <- test_data$count
  pred_correlation <- cor(pred, actual, use = "complete.obs")
  
  return(pred_correlation)
}

# Apply function across each fold
cor_results <- lapply(unique(freezing_counts_data$.folds), function(fold) {
  train_data <- freezing_counts_data[freezing_counts_data$.folds != fold, ]
  test_data <- freezing_counts_data[freezing_counts_data$.folds == fold, ]
  fit_and_evaluate_model(train_data, test_data)
})

freezing_counts_data$`animal:unit` <- with(freezing_counts_data, paste(animal, unit, sep = ":"))

print(table(freezing_counts_data$`animal:unit`))

# Create folds without specifying 'animal' as id_col
set.seed(123)  # For reproducibility
freezing_counts_data <- groupdata2::fold(freezing_counts_data, k = 5, cat_col = 'animal:unit')

# Check the distribution of folds
table(freezing_counts_data$.folds)


fit_and_evaluate_model <- function(train_data, test_data) {
  model <- glmmTMB(count ~ group * period_type * neuron_type * scaled_percent_freezing + 
                     (1|animal:unit:period) + (1|animal:unit),
                   family = nbinom1, data = train_data)
  
  # Predicting on test data
  pred <- predict(model, newdata = test_data, type = "response")
  
  # Calculate correlation between predicted and actual counts
  actual <- test_data$count
  pred_correlation <- cor(pred, actual, use = "complete.obs")
  
  return(pred_correlation)
}

# Apply function across each fold
cor_results <- lapply(unique(freezing_counts_data$.folds), function(fold) {
  train_data <- freezing_counts_data[freezing_counts_data$.folds != fold, ]
  test_data <- freezing_counts_data[freezing_counts_data$.folds == fold, ]
  fit_and_evaluate_model(train_data, test_data)
})

mean_correlation <- mean(unlist(cor_results))
print(mean_correlation)




                                  