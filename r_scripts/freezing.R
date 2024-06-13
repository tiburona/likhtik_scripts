library(dplyr)
library(ggplot2)
library(glmmTMB)
library(emmeans)





just_freezing_csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/percent_freezing', 'freezing.csv', sep='/')
just_freezing_data <- read.csv(just_freezing_csv, comment.char="#")


factor_vars <- c('animal', 'group', 'period_type')
just_freezing_data[factor_vars] <- lapply(just_freezing_data[factor_vars], factor)


# Filter data for pretone and tone period types
just_freezing_data <- subset(just_freezing_data, period_type %in% c("pretone", "tone"))

just_freezing_data <- just_freezing_data[!is.na(just_freezing_data$percent_freezing), ]

just_freezing_data$proportion <- just_freezing_data$percent_freezing/100
just_freezing_data$percent_freezing_adj <- pmin(pmax(periods_1$proportion, 0.0001), 0.9999)


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
  scale_x_continuous(breaks = 1:5, labels = 1:5) +
  scale_color_manual(values = c("pretone" = "pink", "tone" = "green")) +
  labs(x = "Period", y = "Percent Freezing", color = "Period Type") +
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

plot(residuals(period_1_model_beta) ~ fitted(period_1_model_beta))
abline(h = 0, col = "red")

emmip_plot <- emmip(period_1_model_beta, group ~ period_type, type = "response")

emmip_plot <- emmip_plot + 
  scale_color_manual(values = c("control" = "purple", "defeat" = "orange")) +
  labs(title = "Period 1 Freezing Behavior",
       x = "Period Type",
       y = "Predicted Proportion Freezing",
       color = "Group") +
  theme_minimal()

print(emmip_plot)


# First, calculate the mean avg_percent_freezing for each group and period_type
periods_summary <- periods_1 %>%
  group_by(group, period_type) %>%
  summarise(avg_percent_freezing = mean(percent_freezing, na.rm = TRUE), .groups = "drop")

# Now plot using ggplot
ggplot(periods_summary, aes(x = period_type, y = avg_percent_freezing, fill = group)) +
  geom_col(position = position_dodge()) +
  geom_point(data = periods_1, aes(x = period_type, y = percent_freezing, group = period_type), 
             position = position_dodge(width = 0.9), color = "gray", size = 3, alpha = 0.6) +
  scale_fill_manual(values = c("control" = "purple", "defeat" = "orange")) +
  facet_wrap(~ group, scales = "free_x") +
  labs(title = "Period 1 Percent Freezing by Period Type and Group",
       x = "Period Type",
       y = "Average Percent Freezing") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
