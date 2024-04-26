library(lme4)
library(dplyr)
library(ggplot2)
library(emmeans)
library(rlang)
library(tidyr)
library(readxl)
library(stringr)

csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/mrl', 'psth_mrl_freezing.csv', sep='/')
data <- read.csv(csv, comment.char="#") 
read_metadata(csv)

factor_vars <- c('animal', 'group', 'period_type', 'unit', 'neuron_type')
data[factor_vars] <- lapply(data[factor_vars], factor)


freezing_rate_data <- data %>%
  filter(time_bin < 60) %>%
  group_by(period, group, period_type, neuron_type, unit, animal, event) %>%
  summarise(
    rate = mean(rate, na.rm = TRUE),
    percent_freezing = mean(percent_freezing, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )

freezing_rate_model <- lmer(rate ~ group*period_type*percent_freezing*neuron_type + (1|animal/unit/period), data=freezing_rate_data)
summary(freezing_rate_model)

predictions_data <- create_predictions_data(freezing_rate_data, freezing_rate_model, 'percent_freezing')

graph_predictions(predictions_data, 'percent_freezing', 'predicted', "Percent Freezing", "Predicted Rate", num_vars=4)





# Filter data for pretone and tone period types
subset_data <- subset(data, period_type %in% c("pretone", "tone"))

subset_data <- data[!is.na(subset_data$percent_freezing), ]


# Calculate average percent freezing for each period, group, and period type
average_data <- subset_data %>%
  group_by(group, period, period_type) %>%
  summarise(avg_percent_freezing = mean(percent_freezing))
  .groups = "drop"

# Create the plot with facet_grid
ggplot(average_data, aes(x = period, y = avg_percent_freezing, group = period_type, color = period_type)) +
  geom_line() +
  facet_grid(group ~ ., scales = "free_y") +
  scale_x_continuous(breaks = 1:5, labels = 1:5) +
  scale_color_manual(values = c("pretone" = "red", "tone" = "blue")) +
  labs(x = "Period", y = "Average Percent Freezing", color = "Period Type") +
  theme_minimal()


freezing_data <- subset_data %>%
  group_by(group, period, period_type, animal) %>%
  summarise(avg_percent_freezing = mean(percent_freezing))
  .groups = "drop"

freezing_model <- lmer(avg_percent_freezing ~ group*period_type + (1|animal), data=freezing_data)

summary(freezing_model)

# Generate graphs for each model
plot <- emmip(freezing_model, group ~ period_type, CIs = FALSE) + 
  labs(y = paste("Predicted Freezing")) +
  scale_color_manual(values = c("control" = "green", "defeat" = "orange"))

periods_1_2 <- subset(freezing_data, period < 3)


early_model <- lmer(avg_percent_freezing ~ group*period_type + (1|animal), data=periods_1_2)

summary(early_model)


periods_4_5 <- subset(freezing_data, period < 3)
late_model <- lmer(avg_percent_freezing ~ group*period_type + (1|animal), data=periods_4_5)
summary(late_model)


freezing_model_period_predictor <- lm(avg_percent_freezing ~ group*period_type*period, data=freezing_data)

summary(freezing_model_period_predictor)


new_data <- expand.grid(group = c("control", "defeat"),
                        period_type = c("pretone", "tone"),
                        period = c(1, 2, 3, 4, 5))

predicted_values <- predict(freezing_model_period_predictor, newdata = new_data)

# Create data frame for plotting
plot_data <- data.frame(new_data, predicted_values)

# Plot predicted values
ggplot(plot_data, aes(x = period, y = predicted_values, color = period_type)) +
  geom_line() +
  facet_wrap(~group) +
  labs(x = "Period", y = "Predicted Avg. Percent Freezing", color = "Period Type") +
  scale_color_manual(values = c("red", "blue")) +
  theme_minimal()


excel_name = "/Users/katie/likhtik/IG_INED_Safety_Recall/Freezing for Katie.xlsx"


# Read the spreadsheet without headers
data <- read_excel(excel_name, sheet = 1, col_names = FALSE)

# Manually create headers from the first two rows
headers <- data %>% 
  slice(1:2) %>%
  unlist() %>%
  as.character()

# Combine headers where necessary
clean_headers <- headers[1,] # This assumes the first row has the main headers
secondary_headers <- headers[2,] # This assumes the second row has sub-headers

# If a main header is blank, replace it with the corresponding sub-header
for(i in seq_along(clean_headers)) {
  if(clean_headers[i] == "") {
    clean_headers[i] <- secondary_headers[i]
  } else {
    # Combine main header and sub-header if both are present
    if(secondary_headers[i] != "") {
      clean_headers[i] <- paste(clean_headers[i], secondary_headers[i], sep = "_")
    }
  }
}

# Replace spaces and special characters in header names
clean_headers <- gsub("[[:space:]]+|\\.", "_", clean_headers)
clean_headers <- tolower(clean_headers)

# Assign the clean headers to the data
names(data) <- clean_headers

# Remove the first two header rows
data <- data[-c(1, 2), ]

# Now proceed with the transformations, starting with renaming the 'ID' and 'Group' columns
data <- data %>%
  rename(animal = `id`, group = `group`) %>%
  mutate(group = tolower(as.character(group)))

# Reshape the data to a long format, etc...
# [The rest of the data transformation steps go here]

# Write the final data to a CSV
write.csv(data_long, "transformed_data.csv", row.names = FALSE)