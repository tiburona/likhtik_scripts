# Load necessary libraries
library(dplyr)
library(tidyr)

# Assuming your dataset is named 'df'
# Replace 'df' with the actual name of your dataset

csv_dir = '/Users/katie/likhtik/IG_INED_Safety_Recall/power'

csv_name = 'kick_tires.csv'

csv_file = paste(csv_dir, csv_name, sep='/')
df <- read.csv(csv_file, comment.char="#") 


subset_data <- df %>%
  filter(block_type %in% c("tone", "pretone"))

# Group by group, frequency_bin, time_bin, and block_type, and calculate the mean
grouped_data <- subset_data %>%
  group_by(group, frequency_bin, block_type, time_bin) %>%
  summarise(mean_power = mean(pl_.0..15._power))

# Pivot the data to create matrices
pivoted_data <- grouped_data %>%
  pivot_wider(names_from = time_bin, values_from = mean_power)

# Separate the data for each group and block_type
defeat_tone_matrix <- pivoted_data %>%
  filter(group == "defeat" & block_type == "tone")

control_tone_matrix <- pivoted_data %>%
  filter(group == "control" & block_type == "tone") 

defeat_pretone_matrix <- pivoted_data %>%
  filter(group == "defeat" & block_type == "pretone") 

control_pretone_matrix <- pivoted_data %>%
  filter(group == "control" & block_type == "pretone") 

defeat_tone_matrix <- defeat_tone_matrix[, !colnames(defeat_tone_matrix) %in% c("group", "block_type", "frequency_bin")]
control_tone_matrix <- control_tone_matrix[, !colnames(control_tone_matrix) %in% c("group", "block_type", "frequency_bin")]
defeat_pretone_matrix <- defeat_pretone_matrix[, !colnames(defeat_pretone_matrix) %in% c("group", "block_type", "frequency_bin")]
control_pretone_matrix <- control_pretone_matrix[, !colnames(control_pretone_matrix) %in% c("group", "block_type", "frequency_bin")]

# Convert the dataframes to matrices
defeat_tone_matrix <- as.matrix(defeat_tone_matrix)
control_tone_matrix <- as.matrix(control_tone_matrix)

defeat_pretone_matrix <- as.matrix(rowMeans(defeat_pretone_matrix))
control_pretone_matrix <- as.matrix(rowMeans(control_pretone_matrix))

# Calculate the difference between tone and pretone matrices for each group
defeat_difference_matrix <- defeat_tone_matrix - matrix(rep(defeat_pretone_matrix, each = 35), ncol = 35, byrow = TRUE)

control_difference_matrix <- defeat_tone_matrix - matrix(rep(control_pretone_matrix, each = 35), ncol = 35, byrow = TRUE)


# Now, you have the "pretone" matrices averaged over time_bin and subtracted from the "tone" matrices for "defeat" and "control" groups.