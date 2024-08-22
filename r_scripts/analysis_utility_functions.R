library(ggplot2)


read_metadata <- function(csv_file) {
  # Read in all lines of the file
  all_lines <- readLines(csv_file)
  
  # Filter lines that start with the comment character
  metadata_lines <- all_lines[grepl("^#", all_lines)]
  
  # Print the metadata lines
  cat(metadata_lines, sep = "\n")
}

create_predictions_data <- function(data, model, continuous_predictor) {
  
  clean_data <- subset(data, !is.na(data[[continuous_predictor]]))
  
  mean_iv <- mean(clean_data[[continuous_predictor]], na.rm = TRUE)
  sd_iv <- sd(clean_data[[continuous_predictor]], na.rm = TRUE)
  
  pred_data <- expand.grid(
    group = levels(clean_data$group),
    period_type = levels(clean_data$period_type),
#    neuron_type = levels(clean_data$neuron_type),
    iv = c(mean_iv - sd_iv, mean_iv, mean_iv + sd_iv)
  )
  
  # Properly name the power variable in the prediction data
  names(pred_data)[names(pred_data) == "iv"] <- continuous_predictor
  
  pred_data$predicted <- predict(model, newdata = pred_data, re.form = NA)
  
  return(pred_data)
}





graph_predictions <- function(data, x, y, xlabel, ylabel, num_vars=4) {
  # Sort data by x
  data <- data[order(data[[x]]), ]
  
  # Checking unique period_type values for troubleshooting
  print(unique(data$period_type))
  p <- ggplot(data, aes_string(x = x, y = y, color = "period_type")) +
      geom_line() + labs(x = xlabel, y = ylabel) + theme_bw() +
      scale_color_manual(values = c("tone" = "green", "pretone" = "pink"))  # Ensure values match those printed
               
               # Modify facets based on number of variables
               if (num_vars == 4) {
                 p <- p + facet_grid("neuron_type ~ group", scales = "free")
               } else {
                 p <- p + facet_grid(". ~ group", scales = "free")
               }
               
               return(p)
}







