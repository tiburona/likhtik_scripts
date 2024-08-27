library(lme4)
library(dplyr)
library(glmmTMB)
library(lmerTest)
library(sjPlot)
library(ggplot2)
library(emmeans)
library(DHARMa)
library(gamlss)
library(caret)
library(boot)



csv_name = 'granger_behavior_power.csv'
granger_csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/power', csv_name, sep='/')
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



theta_1_data <- granger_df %>%

  mutate(
    proportion = percent_freezing / 100,
    freezing_adj = pmin(pmax(proportion, 0.0001), 0.9999)
  ) %>%
  mutate(
    log_forward = log(forward),
    log_backward = log(backward)
  ) %>%
  filter(frequency >= 4 & frequency <= 8) %>%  # Filtering for frequency between 4 and 8
  group_by(animal, period_type, granger_calculator, group, period) %>%
  summarize(
    bla_theta_1_power = mean(bla_theta_1_power),
    pl_theta_1_power = mean(pl_theta_1_power),
    bla_theta_2_power = mean(bla_theta_1_power),
    pl_theta_2_power = mean(pl_theta_2_power),
    freezing_adj = mean(freezing_adj, na.rm = TRUE),
    bla_pl = mean(forward, na.rm = TRUE),
    pl_bla = mean(backward, na.rm = TRUE),
    log_bla_pl = mean(log_forward, na.rm = TRUE),
    log_pl_bla = mean(log_backward, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  mutate(
    scaled_bla_pl = scale(bla_pl),
    scaled_log_bla_pl = scale(log_bla_pl),
    diff = bla_pl - pl_bla,
    diff_logs = log_bla_pl - log_pl_bla,
    freezing_cat = factor(ifelse(freezing_adj > 0.5, "High", "Low"))  # Binarize freezing_adj
  )



theta_2_data <- granger_df %>%
  
  mutate(
    proportion = percent_freezing / 100,
    freezing_adj = pmin(pmax(proportion, 0.0001), 0.9999)
  ) %>%
  mutate(
    log_forward = log(forward),
    log_backward = log(backward)
  ) %>%
  filter(frequency >= 8 & frequency <= 12) %>%  # Filtering for frequency between 4 and 8
  group_by(animal, period_type, granger_calculator, group, period) %>%
  summarize(
    bla_theta_1_power = mean(bla_theta_1_power),
    pl_theta_1_power = mean(pl_theta_1_power),
    bla_theta_2_power = mean(bla_theta_1_power),
    pl_theta_2_power = mean(pl_theta_2_power),
    freezing_adj = mean(freezing_adj, na.rm = TRUE),
    bla_pl = mean(forward, na.rm = TRUE),
    pl_bla = mean(backward, na.rm = TRUE),
    log_bla_pl = mean(log_forward, na.rm = TRUE),
    log_pl_bla = mean(log_backward, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  mutate(
    scaled_bla_pl = scale(bla_pl),
    scaled_log_bla_pl = scale(log_bla_pl),
    diff = bla_pl - pl_bla,
    diff_logs = log_bla_pl - log_pl_bla,
    freezing_cat = factor(ifelse(freezing_adj > 0.5, "High", "Low"))  # Binarize freezing_adj
  )





first_period_theta_1_data <- theta_1_data %>%
  filter(period == 0) 

first_period_theta_2_data <- theta_2_data %>%
  filter(period == 0) 

freezing_granger_theta_1 <- lmer(freezing_adj ~ group * period_type * scaled_log_bla_pl + (1|animal), 
                           data = theta_1_data)

freezing_granger_theta_2 <- lmer(freezing_adj ~ group * period_type * scaled_log_bla_pl + (1|animal), 
                                 data = theta_2_data)

freezing_granger_theta_1_first_period <- lmer(freezing_adj ~ group * period_type * scaled_log_bla_pl + (1|animal), 
                                                                          data = first_period_theta_1_data)

freezing_granger_theta_2_first_period <- lmer(freezing_adj ~ group * period_type * scaled_log_bla_pl + (1|animal), 
                                              data = first_period_theta_2_data)

summary(freezing_granger_theta_1)
summary(freezing_granger_theta_1_first_period)

residuals <- resid(freezing_granger_theta_1)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

set.seed(123)


# Define a function that returns the fixed effects coefficients from the model
fixed_effects_fun <- function(model) {
  fixef(model)
}


# Define a function to print out coefficient name, average value, and confidence intervals concisely with an asterisk if CI does not include 0
print_coef_summary <- function(boot_results, average_coefficients) {
  for (i in 1:length(average_coefficients)) {
    coef_name <- names(average_coefficients)[i]
    avg_value <- round(average_coefficients[i], 4)
    
    # Calculate the confidence interval
    ci <- boot.ci(boot_results, type = "perc", index = i)
    ci_lower <- round(ci$perc[4], 4)
    ci_upper <- round(ci$perc[5], 4)
    
    # Check if CI includes zero and add an asterisk if it doesn't
    significance_marker <- ifelse(ci_lower > 0 | ci_upper < 0, "*", "")
    
    # Print the summary on a single line
    cat(coef_name, ": Avg =", avg_value, significance_marker, ", 95% CI = [", ci_lower, ",", ci_upper, "]\n")
  }
}

predict_with_avg_coefficients <- function(new_data, avg_coef) {
  # Construct the linear predictor using the average coefficients
  linear_predictor <- avg_coef["(Intercept)"] +
    avg_coef["groupdefeat"] * as.numeric(new_data$group == "defeat") +
    avg_coef["period_typetone"] * as.numeric(new_data$period_type == "tone") +
    avg_coef["scaled_log_bla_pl"] * new_data$scaled_log_bla_pl +
    avg_coef["groupdefeat:period_typetone"] * as.numeric(new_data$group == "defeat") * as.numeric(new_data$period_type == "tone") +
    avg_coef["groupdefeat:scaled_log_bla_pl"] * as.numeric(new_data$group == "defeat") * new_data$scaled_log_bla_pl +
    avg_coef["period_typetone:scaled_log_bla_pl"] * as.numeric(new_data$period_type == "tone") * new_data$scaled_log_bla_pl +
    avg_coef["groupdefeat:period_typetone:scaled_log_bla_pl"] * as.numeric(new_data$group == "defeat") * as.numeric(new_data$period_type == "tone") * new_data$scaled_log_bla_pl
  
  return(linear_predictor)
}

# Perform bootstrap
boot_results <- bootMer(freezing_granger_theta_1, fixed_effects_fun, nsim = 1000, type = "parametric")

# Get the bootstrap confidence intervals for each coefficient and print with names
ci_list <- lapply(1:ncol(boot_results$t), function(i) {
  coef_name <- names(fixef(freezing_granger_theta_2))[i]
  ci <- boot.ci(boot_results, type = "perc", index = i)
  cat("Confidence intervals for", coef_name, ":\n")
  print(ci)
  cat("\n")
})

mean_granger <- mean(theta_1_data$scaled_log_bla_pl, na.rm = TRUE)
sd_granger <- sd(theta_1_data$scaled_log_bla_pl, na.rm = TRUE)

new_data <- expand.grid(
  group = c("control", "defeat"),
  period_type = c("tone", "pretone"),
  scaled_log_bla_pl = c(mean_granger - sd_granger, mean_granger, mean_granger + sd_granger),
  animal = unique(theta_1_data$animal)
)

# Compute the average of the bootstrapped coefficients
average_coefficients <- apply(boot_results$t, 2, mean)
names(average_coefficients) <- names(fixef(freezing_granger_theta_1))

# Call the function to print the summary
print_coef_summary(boot_results, average_coefficients)

# Apply the function to the new_data
new_data$predicted_freezing_avg <- predict_with_avg_coefficients(new_data, average_coefficients)

# Calculate the mean prediction across animals
mean_predictions_avg <- new_data %>%
  group_by(group, period_type, scaled_log_bla_pl) %>%
  summarise(mean_predicted_freezing_avg = mean(predicted_freezing_avg, na.rm = TRUE)) %>%
  ungroup()

ggplot(mean_predictions_avg, aes(x = scaled_log_bla_pl, y = mean_predicted_freezing_avg, color = period_type)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +  # Add points to highlight predictions
  facet_wrap(~group) +
  scale_color_manual(values = c("tone" = "green", "pretone" = "pink")) +
  scale_x_continuous(breaks = c(mean_granger - sd_granger, mean_granger, mean_granger + sd_granger),
                     labels = c("-1 SD", "Mean", "+1 SD")) +  # Fix x-axis labels
  labs(
    x = "BLA_PL",
    y = "Mean Predicted Freezing",
    color = "Period Type"
  ) +
  theme_minimal()


# Perform bootstrap
boot_results <- bootMer(freezing_granger_theta_2, fixed_effects_fun, nsim = 1000, type = "parametric")

# Get the bootstrap confidence intervals for each coefficient and print with names
ci_list <- lapply(1:ncol(boot_results$t), function(i) {
  coef_name <- names(fixef(freezing_granger_theta_2))[i]
  ci <- boot.ci(boot_results, type = "perc", index = i)
  cat("Confidence intervals for", coef_name, ":\n")
  print(ci)
  cat("\n")
})

mean_granger <- mean(theta_2_data$scaled_log_bla_pl, na.rm = TRUE)
sd_granger <- sd(theta_2_data$scaled_log_bla_pl, na.rm = TRUE)

new_data <- expand.grid(
  group = c("control", "defeat"),
  period_type = c("tone", "pretone"),
  scaled_log_bla_pl = c(mean_granger - sd_granger, mean_granger, mean_granger + sd_granger),
  animal = unique(theta_1_data$animal)
)

# Compute the average of the bootstrapped coefficients
average_coefficients <- apply(boot_results$t, 2, mean)
names(average_coefficients) <- names(fixef(freezing_granger_theta_1))

# Call the function to print the summary
print_coef_summary(boot_results, average_coefficients)

# Apply the function to the new_data
new_data$predicted_freezing_avg <- predict_with_avg_coefficients(new_data, average_coefficients)

# Calculate the mean prediction across animals
mean_predictions_avg <- new_data %>%
  group_by(group, period_type, scaled_log_bla_pl) %>%
  summarise(mean_predicted_freezing_avg = mean(predicted_freezing_avg, na.rm = TRUE)) %>%
  ungroup()

ggplot(mean_predictions_avg, aes(x = scaled_log_bla_pl, y = mean_predicted_freezing_avg, color = period_type)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +  # Add points to highlight predictions
  facet_wrap(~group) +
  scale_color_manual(values = c("tone" = "green", "pretone" = "pink")) +
  scale_x_continuous(breaks = c(mean_granger - sd_granger, mean_granger, mean_granger + sd_granger),
                     labels = c("-1 SD", "Mean", "+1 SD")) +  # Fix x-axis labels
  labs(
    x = "BLA_PL",
    y = "Mean Predicted Freezing",
    color = "Period Type"
  ) +
  theme_minimal()


# Perform bootstrap
boot_results <- bootMer(freezing_granger_theta_1_first_period, fixed_effects_fun, nsim = 1000, type = "parametric")

# Get the bootstrap confidence intervals for each coefficient and print with names
ci_list <- lapply(1:ncol(boot_results$t), function(i) {
  coef_name <- names(fixef(freezing_granger_theta_1_first_period))[i]
  ci <- boot.ci(boot_results, type = "perc", index = i)
  cat("Confidence intervals for", coef_name, ":\n")
  print(ci)
  cat("\n")
})

mean_granger <- mean(first_period_theta_1_data$scaled_log_bla_pl, na.rm = TRUE)
sd_granger <- sd(first_period_theta_1_data$scaled_log_bla_pl, na.rm = TRUE)

new_data <- expand.grid(
  group = c("control", "defeat"),
  period_type = c("tone", "pretone"),
  scaled_log_bla_pl = c(mean_granger - sd_granger, mean_granger, mean_granger + sd_granger),
  animal = unique(theta_1_data$animal)
)

# Compute the average of the bootstrapped coefficients
average_coefficients <- apply(boot_results$t, 2, mean)
names(average_coefficients) <- names(fixef(freezing_granger_theta_1))

# Call the function to print the summary
print_coef_summary(boot_results, average_coefficients)

# Apply the function to the new_data
new_data$predicted_freezing_avg <- predict_with_avg_coefficients(new_data, average_coefficients)

# Calculate the mean prediction across animals
mean_predictions_avg <- new_data %>%
  group_by(group, period_type, scaled_log_bla_pl) %>%
  summarise(mean_predicted_freezing_avg = mean(predicted_freezing_avg, na.rm = TRUE)) %>%
  ungroup()

ggplot(mean_predictions_avg, aes(x = scaled_log_bla_pl, y = mean_predicted_freezing_avg, color = period_type)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +  # Add points to highlight predictions
  facet_wrap(~group) +
  scale_color_manual(values = c("tone" = "green", "pretone" = "pink")) +
  scale_x_continuous(breaks = c(mean_granger - sd_granger, mean_granger, mean_granger + sd_granger),
                     labels = c("-1 SD", "Mean", "+1 SD")) +  # Fix x-axis labels
  labs(
    x = "BLA_PL",
    y = "Mean Predicted Freezing",
    color = "Period Type"
  ) +
  theme_minimal()

# Perform bootstrap
boot_results <- bootMer(freezing_granger_theta_2_first_period, fixed_effects_fun, nsim = 1000, type = "parametric")

# Get the bootstrap confidence intervals for each coefficient and print with names
ci_list <- lapply(1:ncol(boot_results$t), function(i) {
  coef_name <- names(fixef(freezing_granger_theta_2_first_period))[i]
  ci <- boot.ci(boot_results, type = "perc", index = i)
  cat("Confidence intervals for", coef_name, ":\n")
  print(ci)
  cat("\n")
})

mean_granger <- mean(first_period_theta_2_data$scaled_log_bla_pl, na.rm = TRUE)
sd_granger <- sd(first_period_theta_2_data$scaled_log_bla_pl, na.rm = TRUE)

new_data <- expand.grid(
  group = c("control", "defeat"),
  period_type = c("tone", "pretone"),
  scaled_log_bla_pl = c(mean_granger - sd_granger, mean_granger, mean_granger + sd_granger),
  animal = unique(theta_1_data$animal)
)

# Compute the average of the bootstrapped coefficients
average_coefficients <- apply(boot_results$t, 2, mean)
names(average_coefficients) <- names(fixef(freezing_granger_theta_1))

# Call the function to print the summary
print_coef_summary(boot_results, average_coefficients)

# Apply the function to the new_data
new_data$predicted_freezing_avg <- predict_with_avg_coefficients(new_data, average_coefficients)

# Calculate the mean prediction across animals
mean_predictions_avg <- new_data %>%
  group_by(group, period_type, scaled_log_bla_pl) %>%
  summarise(mean_predicted_freezing_avg = mean(predicted_freezing_avg, na.rm = TRUE)) %>%
  ungroup()

ggplot(mean_predictions_avg, aes(x = scaled_log_bla_pl, y = mean_predicted_freezing_avg, color = period_type)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +  # Add points to highlight predictions
  facet_wrap(~group) +
  scale_color_manual(values = c("tone" = "green", "pretone" = "pink")) +
  scale_x_continuous(breaks = c(mean_granger - sd_granger, mean_granger, mean_granger + sd_granger),
                     labels = c("-1 SD", "Mean", "+1 SD")) +  # Fix x-axis labels
  labs(
    x = "BLA_PL",
    y = "Mean Predicted Freezing",
    color = "Period Type"
  ) +
  theme_minimal()


model_beta <- glmmTMB(freezing_adj ~ group * period_type * scaled_log_bla_pl + (1 | animal),
                               data = theta_1_data,
                               family = list(family="beta", link="logit"))



# Check the coefficients
model_coefficients <- coef(freezing_granger_theta_1)
print(model_coefficients)






hist(theta_1_data$freezing_adj)

residuals <- resid(freezing_granger_theta_1)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

threshold <- 1.345 * sd(residuals)


# Calculate robust weights (example using Huber's function)
robust_weights <- ifelse(abs(residuals) < threshold, 1, threshold / abs(residuals))

robust_freezing_granger_theta_1 <- gamlss(freezing_adj ~ group * period_type * scaled_log_bla_pl + random(animal), 
                                   data = theta_1_data,
                                   weights = robust_weights,
                                   sigma.formula = ~ 1,  
                                   nu.formula = ~ 1,  
                                   family = BEOI())


residuals <- resid(robust_freezing_granger_theta_1)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

mean_granger <- mean(theta_1_data$scaled_log_bla_pl, na.rm = TRUE)
sd_granger <- sd(theta_1_data$scaled_log_bla_pl, na.rm = TRUE)

new_data <- expand.grid(
  group = c("control", "defeat"),
  period_type = c("tone", "pretone"),
  scaled_log_bla_pl = c(mean_granger - sd_granger, mean_granger, mean_granger + sd_granger),
  animal = unique(theta_1_data$animal)
)

# Generate predictions
new_data$predicted_freezing <- predict(freezing_granger_theta_1, newdata = new_data, type = "response")

# Calculate the mean prediction across animals
mean_predictions <- new_data %>%
  group_by(group, period_type, scaled_log_bla_pl) %>%
  summarise(mean_predicted_freezing = mean(predicted_freezing, na.rm = TRUE)) %>%
  ungroup()

ggplot(mean_predictions, aes(x = scaled_log_bla_pl, y = mean_predicted_freezing, color = period_type)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +  # Add points to highlight predictions
  facet_wrap(~group) +
  scale_color_manual(values = c("tone" = "green", "pretone" = "pink")) +
  scale_x_continuous(breaks = c(mean_granger - sd_granger, mean_granger, mean_granger + sd_granger),
                     labels = c("-1 SD", "Mean", "+1 SD")) +  # Fix x-axis labels
  labs(
    x = "BLA_PL",
    y = "Mean Predicted Freezing",
    color = "Period Type"
  ) +
  theme_minimal()


ggplot(data = theta_1_data, aes(x = scaled_log_bla_pl, y = freezing_adj)) +
  geom_point() +
  labs(title = "",
       x = "BLA-PL",
       y = "Freezing") +
  theme_minimal()

freezing_granger_theta_1_first_period <- gamlss(freezing_adj ~ group * period_type * scaled_log_bla_pl + random(animal), 
                                   data = first_period_theta_1_data,
                                   sigma.formula = ~ 1,  
                                   nu.formula = ~ 1,  
                                   family = BEOI())

freezing_granger_theta_1_first_period <- lmer(freezing_adj ~ group * period_type * scaled_log_bla_pl + (1|animal), data = first_period_theta_1_data)

summary(freezing_granger_theta_1_first_period)


hist(first_period_theta_1_data$freezing_adj)

residuals <- resid(freezing_granger_theta_1_first_period)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

freezing_granger_theta_2 <- gamlss(freezing_adj ~ group * period_type * log_bla_pl_theta_1_granger + random(animal), 
                                   data = theta_2_data,
                                   sigma.formula = ~ 1,  
                                   nu.formula = ~ 1,  
                                   family = BEOI())

freezing_granger_theta_2_first_period <- gamlss(freezing_adj ~ group * period_type * log_bla_pl_theta_1_granger + random(animal), 
                                                data = first_period_theta_2_data,
                                                sigma.formula = ~ 1,  
                                                nu.formula = ~ 1,  
                                                family = BEOI())

summary(freezing_granger_theta_1)
summary(freezing_granger_theta_2)


summary(freezing_granger_theta_1_first_period)
summary(freezing_granger_theta_2_first_period)









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



