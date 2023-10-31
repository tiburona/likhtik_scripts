library(lme4)
library(dplyr)
library(ggplot2)
library(emmeans)
library(rlang)

mrl_freezing_csv <- '/Users/katie/likhtik/data/lfp/percent_freezing/spike_power_mrl.csv'
mrl_freezing_data <- read.csv(mrl_power_csv, comment.char="#") 


# Read in all lines of the file
all_lines <- readLines(mrl_freezing_csv)

# Filter lines that start with the comment character
metadata_lines <- all_lines[grepl("^#", all_lines)]

# Print the metadata lines
cat(metadata_lines, sep = "\n")

data_with_mrl <- mrl_freezing_data %>%
  filter(time_bin < 30) %>%
  group_by(period, group, period_type, neuron_type, animal, unit) %>%
  summarise(
    hpc_theta_1_mrl = mean(hpc_theta_1_mrl, na.rm = TRUE),
    bla_theta_1_mrl = mean(bla_theta_1_mrl, na.rm = TRUE),
    pl_theta_1_mrl  = mean(pl_theta_1_mrl, na.rm = TRUE),
    percent_freezing = mean(percent_freezing, na.rm = TRUE),
    .groups = "drop"
  )

make_model <- function(iv, data) {
  formula <- as.formula(paste("percent_freezing ~", iv, " * period_type * group + (1|animal/unit)"))
  model <- lmer(formula, data = data)
  print(summary(model))
  
  mean_iv <- mean(data[[iv]], na.rm = TRUE)
  sd_iv <- sd(data[[iv]], na.rm = TRUE)
  
  predictions_data <- create_predictions_data(data, model, iv)
  plot <- graph_predictions(predictions_data, iv, "predicted", iv, paste("Predicted", dv))
  
  return(list(data = predictions_data, model = model, plot = plot))
}



variables <- c("hpc_theta_1_mrl", "bla_theta_1_mrl", "pl_theta_1_mrl")



results <- list()
for (i in seq_along(variables)) {
  iv <- variables[i]
  cat("Model and plot for IV:", iv, "\n")
  results[iv] <- make_model(iv, data_with_mrl)
}


summary(predictions_data[predictions_data$group == "stressed", ])
