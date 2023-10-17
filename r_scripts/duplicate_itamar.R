
# Load necessary libraries
library(ggplot2)
library(dplyr)

csv_name = 'power_hpc_theta_1_power_hpc_theta_2_hpc_replication_power_by_period.csv'
csv_dir = '/Users/katie/likhtik/data/lfp/power'

csv_file = paste(csv_dir, csv_name, sep='/')
data <- read.csv(csv_file, comment.char="#") 

# Filter data based on your criteria
filtered_data <- data[data$period %in% c(0, 1) & 
                        data$group %in% c('control', 'stressed') & 
                        data$period_type %in% c('tone', 'pretone'), ]

# Adjust period values for x-axis labeling
filtered_data$period <- ifelse(filtered_data$period == 0, 1, 2)

# Calculate means
mean_data <- filtered_data %>%
  group_by(period, group, period_type) %>%
  summarise(mean_hpc_theta_1_power = mean(hpc_theta_1_power, na.rm = TRUE))

# Plot
ggplot(mean_data, aes(x = period, y = mean_hpc_theta_1_power, color = period_type, shape = period_type)) +
  geom_point(aes(shape = period_type, color = period_type), size = 3) +
  geom_line(aes(group = period_type)) +
  scale_shape_manual(values = c("pretone" = 15, "tone" = 16)) + # 15 is for squares and 16 is for circles
  scale_color_manual(values = c("pretone" = "gray", "tone" = "blue")) +
  labs(x = "Period", y = "Mean HPC Theta 1 Power") +
  facet_wrap(~ group, ncol = 1) + # Separate panels for each level of 'group'
  theme_minimal()
