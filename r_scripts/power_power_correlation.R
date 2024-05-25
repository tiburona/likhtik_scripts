library(dplyr)
library(mgcv)
library(lme4)
library(ggplot2)
library(emmeans)
library(rlang)
library(tidyr)
library(lmerTest)


max_correlation_csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/correlation', 'lag_of_max_correlation.csv', sep='/')
max_corr_data <- read.csv(max_correlation_csv, comment.char="#") 
factor_vars <- c('animal', 'group', 'period_type')
data[factor_vars] <- lapply(max_corr_data[factor_vars], factor)

average_max_corr_data <- max_corr_data %>% 
  group_by(group, period_type, animal) %>%
  summarise(bla_pl_theta_1_correlation = mean(bla_pl_theta_1_correlation, na.rm=TRUE))
.groups = "drop"


bla_pl_control_tone = subset(average_max_corr_data, average_max_corr_data$group == 'control' & average_max_corr_data$period_type == 'tone' & !is.na(average_max_corr_data$bla_pl_theta_1_correlation))
bla_pl_control_pretone = subset(average_max_corr_data, average_max_corr_data$group == 'control' & average_max_corr_data$period_type == 'pretone' & !is.na(average_max_corr_data$bla_pl_theta_1_correlation))
bla_pl_defeat_tone = subset(average_max_corr_data, average_max_corr_data$group == 'defeat' & average_max_corr_data$period_type == 'tone' & !is.na(average_max_corr_data$bla_pl_theta_1_correlation))
bla_pl_defeat_pretone = subset(average_max_corr_data, average_max_corr_data$group == 'defeat' & average_max_corr_data$period_type == 'pretone' & !is.na(average_max_corr_data$bla_pl_theta_1_correlation))


wilcox.test(bla_pl_control_pretone$bla_pl_theta_1_correlation, mu=0, alternative="two.sided")
wilcox.test(bla_pl_control_tone$bla_pl_theta_1_correlation, mu=0, alternative="two.sided")
wilcox.test(bla_pl_defeat_pretone$bla_pl_theta_1_correlation, mu=0, alternative="two.sided")
wilcox.test(bla_pl_defeat_tone$bla_pl_theta_1_correlation, mu=0, alternative="two.sided")


m <- lmer(bla_pl_theta_1_correlation ~ group * period_type + (1|animal), data=max_corr_data)
summary(m)

plot <- emmip(m, group ~ period_type, CIs = FALSE) + 
  labs(y = paste("Lag of Max Correlation")) +
  scale_color_manual(values = c("control" = "green", "defeat" = "orange"))

plot(resid(m) ~ fitted(m))

qqnorm(resid(m))
qqline(resid(m), col = "red")

ranef_plot <- ranef(m, condVar = TRUE)
plot(ranef_plot)



correlation_csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/correlation', 'correlation.csv', sep='/')
corr_data <- read.csv(correlation_csv, comment.char="#") 

corr_data <- corr_data[!is.na(corr_data[['bla_pl_theta_1_correlation']]), ]

factor_vars <- c('animal', 'group', 'period_type', 'correlation_calculator')
corr_data[factor_vars] <- lapply(corr_data[factor_vars], factor)
corr_data$time_squared <- corr_data$time^2

model <- lmer(bla_pl_theta_1_correlation ~ time_squared + time* group * period_type + (1|animal), data = corr_data)
summary(model)







# Generate the emmip plot
emmip_plot <- emmip(model, period_type ~ time | group, 
                    at = list(time = c(-.1, -.05, 0, .05, .1)),
                    aes(color = period_type))  # Using aes to map color

# Define the color mapping directly related to factor levels
colors_vector <- c("pretone" = "#E75480", "tone" = "#76BD4E")

# Add the manual color scale to the plot
emmip_plot <- emmip_plot + 
  scale_color_manual(values = colors_vector,
                     name = "Period Type",
                     labels = names(colors_vector)) +
  labs(x = "Lags", y = "Linear component of predicted BLA-PL correlation")

# Print the plot
print(emmip_plot)



