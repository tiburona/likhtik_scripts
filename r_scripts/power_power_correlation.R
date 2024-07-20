library(dplyr)
library(mgcv)
library(lme4)
library(ggplot2)
library(emmeans)
library(rlang)
library(tidyr)
library(lmerTest)

### No positive results using "lag of max corr" strategy"

max_correlation_csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/lag_of_max_correlation', 'lag_of_max_correlations.csv', sep='/')
max_corr_data <- read.csv(max_correlation_csv, comment.char="#") 
factor_vars <- c('animal', 'group', 'period_type')
max_corr_data[factor_vars] <- lapply(max_corr_data[factor_vars], factor)
max_corr_data$bla_pl_theta_1_lag = max_corr_data$bla_pl_theta_1_lag_of_max_correlation - 201

average_max_corr_data <- max_corr_data %>% 
  group_by(group, period_type, animal) %>%
  summarise(bla_pl_theta_1_lag = mean(bla_pl_theta_1_lag, na.rm=TRUE))
.groups = "drop"


bla_pl_control_tone = subset(average_max_corr_data, average_max_corr_data$group == 'control' & average_max_corr_data$period_type == 'tone' & !is.na(average_max_corr_data$bla_pl_theta_1_lag))
bla_pl_control_pretone = subset(average_max_corr_data, average_max_corr_data$group == 'control' & average_max_corr_data$period_type == 'pretone' & !is.na(average_max_corr_data$bla_pl_theta_1_lag))
bla_pl_defeat_tone = subset(average_max_corr_data, average_max_corr_data$group == 'defeat' & average_max_corr_data$period_type == 'tone' & !is.na(average_max_corr_data$bla_pl_theta_1_lag))
bla_pl_defeat_pretone = subset(average_max_corr_data, average_max_corr_data$group == 'defeat' & average_max_corr_data$period_type == 'pretone' & !is.na(average_max_corr_data$bla_pl_theta_1_lag))


wilcox.test(bla_pl_control_pretone$bla_pl_theta_1_lag, mu=0, alternative="two.sided")
wilcox.test(bla_pl_control_tone$bla_pl_theta_1_lag, mu=0, alternative="two.sided")
wilcox.test(bla_pl_defeat_pretone$bla_pl_theta_1_lag, mu=0, alternative="two.sided")
wilcox.test(bla_pl_defeat_tone$bla_pl_theta_1_lag, mu=0, alternative="two.sided")

transformed_data <- max_corr_data %>%
  mutate(lag_bin = floor(bla_pl_theta_1_lag_of_max_correlation / 10)) %>%
  group_by(group, animal, period_type, lag_bin) %>%
  summarise(count = n(), .groups = 'drop')


lag_model_poisson <- glmer(count ~ group * period_type * lag_bin + (1|animal), 
                                  family = poisson(link = "log"), 
                                  data = transformed_data)

linear.model <- lmer(bla_pl_theta_1_lag_of_max_correlation ~ group + period_type + (1|animal), data=max_corr_data)
summary(linear.model)

max_corr_data$log_bla_pl_theta_1 <- log(max_corr_data$bla_pl_theta_1_lag_of_max_correlation + .5) 
max_corr_data$sqrt_bla_pl_theta_1 <- (max_corr_data$bla_pl_theta_1_lag_of_max_correlation)**.5
log_model <- lmer(log_bla_pl_theta_1 ~ group + period_type + (1|animal), data=max_corr_data)
sqrt_model <- lmer(sqrt_bla_pl_theta_1 ~ group + period_type + (1|animal), data=max_corr_data)



residuals <- resid(sqrt_model)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

plot(residuals ~ fitted(linear.model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

hist(residuals, main = "Histogram of Residuals")


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


# Starting from here, code for analyzing and graphing the data using the actual
# correlation as DV


correlation_csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/correlation', 'correlation.csv', sep='/')
corr_data <- read.csv(correlation_csv, comment.char="#") 

corr_data <- corr_data[!is.na(corr_data[['bla_pl_theta_1_correlation']]), ]

factor_vars <- c('animal', 'group', 'period_type', 'correlation_calculator')
corr_data[factor_vars] <- lapply(corr_data[factor_vars], factor)
corr_data$time_squared <- corr_data$time^2

model_glmmTMB <- glmmTMB(bla_pl_theta_1_correlation ~ time_squared + time* group * period_type + (1|animal), data = corr_data)

summary(model_glmmTMB)

simulationOutput <- simulateResiduals(fittedModel = model_glmmTMB, plot = TRUE)

# Generate the emmip plot
emmip_plot <- emmip(model_glmmTMB, period_type ~ time | group, 
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



