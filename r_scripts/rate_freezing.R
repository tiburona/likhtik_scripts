library(lme4)
library(dplyr)
library(ggplot2)
library(emmeans)
library(rlang)

deviation_csv = paste('/Users/katie/likhtik/data/lfp/percent_freezing', 'theta_power_deviations.csv', sep='/')
deviation_data <- read.csv(deviation_csv, comment.char="#") 


# Read in all lines of the file
all_lines <- readLines(deviation_csv)

# Filter lines that start with the comment character
metadata_lines <- all_lines[grepl("^#", all_lines)]

# Print the metadata lines
cat(metadata_lines, sep = "\n")



freezing_rate_data <- deviation_data %>%
  filter(time_bin < 30) %>%
  group_by(period, group, period_type, neuron_type, unit, animal, trial) %>%
  summarise(
    rate = mean(rate, na.rm = TRUE),
    percent_freezing = mean(percent_freezing, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )

freezing_rate_model <- lmer(rate ~ group*period_type*percent_freezing*neuron_type + (1|animal/unit), data=freezing_rate_data)
summary(freezing_rate_model)

predictions_data <- create_predictions_data(freezing_rate_data, freezing_rate_model, 'percent_freezing')

graph_predictions(predictions_data, 'percent_freezing', 'predicted', "Percent Freezing", "Predicted Rate", num_vars=4)
