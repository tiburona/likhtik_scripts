library(dplyr)
library(ggplot2)
library(glmmTMB)
library(emmeans)
library(zoib)
library(gamlss)





just_freezing_csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/percent_freezing', 'freezing.csv', sep='/')
just_freezing_data <- read.csv(just_freezing_csv, comment.char="#")


factor_vars <- c('animal', 'group', 'period_type')
just_freezing_data[factor_vars] <- lapply(just_freezing_data[factor_vars], factor)


# Filter data for pretone and tone period types
just_freezing_data <- subset(just_freezing_data, period_type %in% c("pretone", "tone"))

just_freezing_data <- just_freezing_data[!is.na(just_freezing_data$percent_freezing), ]

just_freezing_data$proportion <- just_freezing_data$percent_freezing/100
just_freezing_data$percent_freezing_adj <- pmin(pmax(just_freezing_data$proportion, 0.0001), 0.9999)


# Calculate average percent freezing for each period, group, and period type
average_data <- just_freezing_data %>% 
  group_by(group, period, period_type, animal) %>%
  group_by(group, period, period_type) %>%
  summarise(percent_freezing = mean(percent_freezing, na.rm=TRUE))
.groups = "drop"

print(summary(average_data))



# Create the plot with facet_grid
ggplot(average_data, aes(x = period, y = percent_freezing, group = period_type, color = period_type)) +
  geom_line() +
  facet_grid(group ~ ., scales = "free_y") +
  scale_x_continuous(breaks = 0:4, labels = 1:5) +  # Adjust breaks to 0:4 and labels to 1:5
  scale_color_manual(values = c("pretone" = "#E75480", "tone" = "#76BD4E")) +
  labs(x = "Period", y = "Percent Freezing", color = "") +
  theme_minimal()



model <- lmer(percent_freezing ~ group*period_type + (1|animal), data=just_freezing_data)
summary(model)

periods_1 <- subset(just_freezing_data, period < 1)


period_1_model <- lmer(percent_freezing ~ group*period_type + (1|animal), data=periods_1)

plot(residuals(period_1_model) ~ fitted(period_1_model))
abline(h = 0, col = "red")


model_beta <- glmmTMB(percent_freezing_adj ~ group * period_type + poly(period, 2) + (1 | animal),
                      data = just_freezing_data,
                      family = list(family="beta", link="logit"))

period_1_model_beta <- glmmTMB(percent_freezing_adj ~ group * period_type + (1 | animal),
                      data = periods_1,
                      family = list(family="beta", link="logit"))

residuals <- resid(period_1_model_beta)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

plot(residuals ~ fitted(bla_theta_1_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

hist(residuals, main = "Histogram of Residuals")



model_gamlss <- gamlss(percent_freezing_adj ~ group * period_type + random(animal),
                       sigma.formula = ~ 1,  
                       nu.formula = ~ 1,  
                       family = BEOI(),  
                       data = periods_1)

summary(model_gamlss)

residuals <- resid(model_gamlss)
qqnorm(resid(model_gamlss))
qqline(resid(model_gamlss))

hist(residuals)




# Step 1: Create balanced prediction data
new_data <- expand.grid(
  group = levels(periods_1$group),
  period_type = levels(periods_1$period_type),
  animal = levels(periods_1$animal)  # Include all levels of the random effect
)

# Step 2: Predict for each combination
new_data$predicted <- predict(model_gamlss, newdata = new_data, type = "response")

# Step 3: Average predictions for each fixed effect combination

average_predictions <- new_data %>%
  group_by(group, period_type) %>%
  summarize(average_predicted = mean(predicted, na.rm = TRUE))

# Step 4: Plot or analyze the averaged predictions
print(average_predictions)


plot_average <- ggplot(average_predictions, aes(x = period_type, y = average_predicted, 
                                                color = group, group = group)) +
  geom_point() +  # Add points to the plot
  geom_line() +   # Connect points with lines to show trends within each group
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354")) +  # Customize colors
  labs(title = "Predicted Proportion of Period 1 Freezing",
       x = "",
       y = "Predicted Proportion Freezing",
       color = "") +
  theme_minimal()  # Use a minimal theme for a clean look

# Print the plot
print(plot_average)



# First, calculate the mean avg_percent_freezing for each group and period_type
periods_summary <- periods_1 %>%
  group_by(group, period_type) %>%
  summarise(avg_percent_freezing = mean(percent_freezing, na.rm = TRUE), .groups = "drop")

# Now plot using ggplot
ggplot(periods_summary, aes(x = period_type, y = avg_percent_freezing, fill = group)) +
  geom_col(position = position_dodge()) +
  geom_point(data = periods_1, aes(x = period_type, y = percent_freezing, group = period_type), 
             position = position_dodge(width = 0.9), color = "gray", size = 3, alpha = 0.6) +
  scale_fill_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354")) +
  facet_wrap(~ group, scales = "free_x") +
  labs(title = "Period 1 Percent Freezing",
       x = "",
       y = "Percent Freezing",
       color = "") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
