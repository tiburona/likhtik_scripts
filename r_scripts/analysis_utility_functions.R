library(ggplot2)


read_metadata <- function(csv_file) {
  # Read in all lines of the file
  all_lines <- readLines(csv_file)
  
  # Filter lines that start with the comment character
  metadata_lines <- all_lines[grepl("^#", all_lines)]
  
  # Print the metadata lines
  cat(metadata_lines, sep = "\n")
}

create_predictions_data <- function(data, model, continuous_predictor, num_vars = 4) {
  
  
  # Check if continuous_predictor is a valid column name
  if (!continuous_predictor %in% names(data)) {
    stop("The continuous predictor column was not found in the data frame.")
  }
  
  # Compute mean and standard deviation of continuous predictor
  mean_pred <- mean(data[[continuous_predictor]], na.rm = TRUE)
  sd_pred <- sd(data[[continuous_predictor]], na.rm = TRUE)
  
  # Create new data frame for predictions
  if (num_vars == 4) {
    new_data <- expand.grid(
      mean_predictor = c(mean_pred - sd_pred, mean_pred, mean_pred + sd_pred),
      group = unique(data$group),
      period_type = unique(data$period_type),
      neuron_type = unique(data$neuron_type),
      stringsAsFactors = FALSE
    )
  } else if (num_vars == 3) {
    new_data <- expand.grid(
      mean_predictor = c(mean_pred - sd_pred, mean_pred, mean_pred + sd_pred),
      group = unique(data$group),
      period_type = unique(data$period_type),
      stringsAsFactors = FALSE
    )
  } else {
    stop("num_vars must be either 3 or 4.")
  }
  
  # Add continuous predictor to new data frame
  new_data[[continuous_predictor]] <- new_data$mean_predictor
  
  # Ensure new_data has all required variables
  model_vars <- all.vars(formula(model))
  for (var in model_vars) {
    if (!var %in% names(new_data) && var %in% names(data)) {
      new_data[[var]] <- mean(data[[var]], na.rm = TRUE)
    }
  }
  
  # Drop 'animal' if it exists in the final data set
  new_data$animal <- NULL
  new_data$unit <- NULL
  
  # Make predictions
  new_data$predicted <- predict(model, newdata = new_data, re.form = NA)
  
  return(new_data)
}





graph_predictions <- function(data, x, y, xlabel, ylabel, num_vars=4) {
  # Sort data by x
  data <- data[order(data[[x]]), ]
  
  p <- ggplot(data, aes_string(x = x, y = y, color = "period_type")) +
    geom_line() +
    labs(x = xlabel, y = ylabel) +
    theme_bw()
  
  if (num_vars == 4) {
    p <- p + facet_grid("neuron_type ~ group", scales = "free")
  } else {
    p <- p + facet_grid(". ~ group", scales = "free")
  }
  
  return(p)
}





