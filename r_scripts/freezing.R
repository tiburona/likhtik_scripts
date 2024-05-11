library(dplyr)
library(ggplot2)



just_freezing_csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/percent_freezing', 'freezing.csv', sep='/')
just_freezing_data <- read.csv(just_freezing_csv, comment.char="#")


factor_vars <- c('animal', 'group', 'period_type')
just_freezing_data[factor_vars] <- lapply(just_freezing_data[factor_vars], factor)


# Filter data for pretone and tone period types
just_freezing_data <- subset(just_freezing_data, period_type %in% c("pretone", "tone"))

just_freezing_data <- just_freezing_data[!is.na(just_freezing_data$percent_freezing), ]


# Calculate average percent freezing for each period, group, and period type
average_data <- just_freezing_data %>% 
  group_by(group, period, period_type, animal) %>%
  group_by(group, period, period_type) %>%
  summarise(avg_percent_freezing = mean(percent_freezing, na.rm=TRUE))
.groups = "drop"

print(summary(average_data))


# Create the plot with facet_grid
ggplot(average_data, aes(x = period, y = avg_percent_freezing, group = period_type, color = period_type)) +
  geom_line() +
  facet_grid(group ~ ., scales = "free_y") +
  scale_x_continuous(breaks = 1:5, labels = 1:5) +
  scale_color_manual(values = c("pretone" = "red", "tone" = "blue")) +
  labs(x = "Period", y = "Average Percent Freezing", color = "Period Type") +
  theme_minimal()

periods_1 <- subset(just_freezing_data, period < 1)

# First, calculate the mean avg_percent_freezing for each group and period_type
periods_summary <- periods_1 %>%
  group_by(group, period_type) %>%
  summarise(avg_percent_freezing = mean(percent_freezing, na.rm = TRUE), .groups = "drop")

# Now plot using ggplot
ggplot(periods_summary, aes(x = period_type, y = avg_percent_freezing, fill = group)) +
  geom_col(position = position_dodge()) +
  geom_point(data = periods_1, aes(x = period_type, y = percent_freezing, group = period_type), 
             position = position_dodge(width = 0.9), color = "gray", size = 3, alpha = 0.6) +
  scale_fill_manual(values = c("control" = "green", "defeat" = "orange")) +
  facet_wrap(~ group, scales = "free_x") +
  labs(title = "Period 1 Percent Freezing by Period Type and Group",
       x = "Period Type",
       y = "Average Percent Freezing") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
