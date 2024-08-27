library(lme4)
library(dplyr)
library(glmmTMB)
library(lmerTest)
library(sjPlot)
library(ggplot2)
library(emmeans)
library(DHARMa)
library(gamlss)


csv_name = 'granger_behavior.csv'
granger_csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/percent_freezing', csv_name, sep='/')
granger_df <- read.csv(granger_csv, comment.char="#") 


# Convert animal, group, and period_type to factors
factor_vars <- c('animal', 'group', 'period_type', 'granger_calculator')
granger_df[factor_vars] <- lapply(granger_df[factor_vars], factor) 
# Example data frame


# Find and rename the column
granger_df <- granger_df %>%
  rename_with(
    .fn = ~ "forward",
    .cols = contains("forward")
  ) %>%
  rename_with(
    .fn = ~ "backward",
    .cols = contains("backward")
  )

# df1 <- granger_df[!is.na(granger_df$forward), ]
# 
# # Step 2: Separate the rows where Column2 has data
# df2 <- granger_df[!is.na(granger_df$backward), ]
# 
# # Step 3: Merge the two data frames based on the key columns
# merged_df <- merge(df1, df2, 
#                    by = c('frequency', 'frequency_bin', 'period_type', 'animal', 
#                           'group', 'granger_calculator', 'period'), all = TRUE)
# 
# # This will create columns named 'Column1.x' and 'Column2.y'. 
# # Step 4: Rename columns if necessary
# merged_df <- merged_df %>%
#   rename(forward = forward.x, backward = backward.y)

# Just the first period, theta 1

theta_2_data <- granger_df %>%
  # mutate(
  #   scaled_length = (length - min(length, na.rm = TRUE)) / (max(length, na.rm = TRUE) - min(length, na.rm = TRUE)),  # Min-max scaling
  #   positive_scaled_length = scaled_length + 0.1  # Ensuring all values are positive by adding 0.1
  # ) %>%
  mutate(
    proportion = percent_freezing / 100,
    freezing_adj = pmin(pmax(proportion, 0.0001), 0.9999)
  ) %>%
  mutate(
    log_forward = log(forward),
    log_backward = log(backward)
  ) %>%
  filter(frequency >= 4 & frequency <= 8) %>%  # Filtering for frequency between 4 and 8
  #filter(period == 0) %>%
  group_by(animal, period_type, granger_calculator, group, period) %>%
  summarize(
    bla_theta_1_power = mean(bla_theta_1_power),
    pl_theta_1_power = mean(pl_theta_1_power),
    bla_theta_2_power = mean(bla_theta_1_power),
    pl_theta_2_power = mean(pl_theta_2_power),
    freezing_adj = mean(freezing_adj, na.rm = TRUE),
    bla_pl_theta_1_granger = mean(forward, na.rm = TRUE),
    pl_bla_theta_1_granger = mean(backward, na.rm = TRUE),
    log_bla_pl_theta_1_granger = mean(log_forward, na.rm = TRUE),
    log_pl_bla_theta_1_granger = mean(log_backward, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  mutate(
    diff = bla_pl_theta_1_granger - pl_bla_theta_1_granger,
    diff_logs = log_bla_pl_theta_1_granger - log_pl_bla_theta_1_granger)


theta_2_data <- granger_df %>%
  # mutate(
  #   scaled_length = (length - min(length, na.rm = TRUE)) / (max(length, na.rm = TRUE) - min(length, na.rm = TRUE)),  # Min-max scaling
  #   positive_scaled_length = scaled_length + 0.1  # Ensuring all values are positive by adding 0.1
  # ) %>%
  mutate(
    proportion = percent_freezing / 100,
    freezing_adj = pmin(pmax(proportion, 0.0001), 0.9999)
  ) %>%
  mutate(
    log_forward = log(forward),
    log_backward = log(backward)
  ) %>%
  filter(frequency >= 8 & frequency <= 12) %>%  # Filtering for frequency between 4 and 8
  #filter(period == 0) %>%
  group_by(animal, period_type, granger_calculator, group, period) %>%
  summarize(
    bla_theta_1_power = mean(bla_theta_1_power),
    pl_theta_1_power = mean(pl_theta_1_power),
    bla_theta_2_power = mean(bla_theta_1_power),
    pl_theta_2_power = mean(pl_theta_2_power),
    freezing_adj = mean(freezing_adj, na.rm = TRUE),
    bla_pl_theta_1_granger = mean(forward, na.rm = TRUE),
    pl_bla_theta_1_granger = mean(backward, na.rm = TRUE),
    log_bla_pl_theta_1_granger = mean(log_forward, na.rm = TRUE),
    log_pl_bla_theta_1_granger = mean(log_backward, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  mutate(
    diff = bla_pl_theta_1_granger - pl_bla_theta_1_granger,
    diff_logs = log_bla_pl_theta_1_granger - log_pl_bla_theta_1_granger)
  
theta_1_model <- lmer(diff_logs ~ group * period_type  + (1|animal), data = theta_1_data)
summary(theta_1_model)

emmip(theta_1_model, group ~ period_type, CIs = FALSE) + 
  labs(x = '', y = paste("BLA Leads Advantage")) +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))

theta_2_model <- lmer(diff_logs ~ group * period_type  + (1|animal), data = theta_2_data)
summary(theta_2_model)

emmip(theta_2_model, group ~ period_type, CIs = FALSE) + 
  labs(x = '', y = paste("BLA Leads Advantage")) +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))

residuals <- resid(theta_2_model)

hist(residuals, main = "Histogram of Residuals")

plot(residuals ~ fitted(theta_1_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")



qqnorm(residuals_no_outlier, main = "Q-Q Plot of Residuals")
qqline(residuals_no_outlier, col = "red")

first_period_theta_1_data <- first_period_theta_1_data %>%
  filter(!is.na(bla_pl_theta_1_granger))

freezing_granger <- gamlss(freezing_adj ~ group * period_type * bla_pl_theta_1_granger + random(animal), 
                           data = first_period_theta_1_data,
                           sigma.formula = ~ 1,  
                           nu.formula = ~ 1,  
                           family = BEOI())




hist(granger_data_bla_pl$bla_pl_theta_1_granger)

residuals <- resid(freezing_granger)

hist(residuals, main = "Histogram of Residuals")

plot(residuals ~ fitted(freezing_granger), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")


# Assuming you have calculated or have the standard deviation and mean of log_bla_pl_theta_1_granger
mean_granger <- mean(granger_data_bla_pl$log_bla_pl_theta_1_granger, na.rm = TRUE)
sd_granger <- sd(granger_data_bla_pl$log_bla_pl_theta_1_granger, na.rm = TRUE)

# Create a new data frame with the combinations you need
new_data <- expand.grid(
  group = c("control", "defeat"),
  period_type = c("tone", "pretone"),
  log_bla_pl_theta_1_granger = c(mean_granger - sd_granger, mean_granger, mean_granger + sd_granger),
  animal = unique(granger_data_bla_pl$animal) # Include a random animal or the most common one if necessary
)

# Generate predictions
new_data$predicted_freezing <- predict(freezing_granger, newdata = new_data, type = "response")

# Assuming you have calculated or have the standard deviation and mean of log_bla_pl_theta_1_granger
mean_granger <- mean(granger_data_bla_pl$log_bla_pl_theta_1_granger, na.rm = TRUE)
sd_granger <- sd(granger_data_bla_pl$log_bla_pl_theta_1_granger, na.rm = TRUE)

# Create a new data frame with the combinations you need, including animals
new_data <- expand.grid(
  group = c("control", "defeat"),
  period_type = c("tone", "pretone"),
  log_bla_pl_theta_1_granger = c(mean_granger - sd_granger, mean_granger, mean_granger + sd_granger),
  animal = unique(granger_data_bla_pl$animal)
)

# Generate predictions
new_data$predicted_freezing <- predict(freezing_granger, newdata = new_data, type = "response")

# Calculate the mean prediction across animals
mean_predictions <- new_data %>%
  group_by(group, period_type, log_bla_pl_theta_1_granger) %>%
  summarise(mean_predicted_freezing = mean(predicted_freezing, na.rm = TRUE)) %>%
  ungroup()

# Create the plot
ggplot(mean_predictions, aes(x = log_bla_pl_theta_1_granger, y = mean_predicted_freezing, color = period_type)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +  # Add points to highlight predictions
  facet_wrap(~group) +
  scale_color_manual(values = c("tone" = "green", "pretone" = "pink")) +
  scale_x_continuous(breaks = c(mean_granger - sd_granger, mean_granger, mean_granger + sd_granger),
                     labels = c("-1 SD", "Mean", "+1 SD")) +  # Fix x-axis labels
  labs(
    x = "log_bla_pl_theta_1_granger",
    y = "Mean Predicted Freezing",
    color = "Period Type"
  ) +
  theme_minimal()



