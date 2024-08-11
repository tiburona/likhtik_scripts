library(lme4)
library(dplyr)
library(glmmTMB)
library(lmerTest)
library(sjPlot)
library(ggplot2)
library(emmeans)
library(DHARMa)
library(gamlss)



# Create a data frame from a CSV file
csv_name = 'power_demo.csv'
power_csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/power', csv_name, sep='/')
power_df <- read.csv(power_csv, comment.char="#") 

# View the beginning and end of your data frame

head(power_df, 20)
tail(power_df, 20)

# Convert animal, group, and period_type to factors
factor_vars <- c('animal', 'group', 'period_type')
power_df[factor_vars] <- lapply(power_df[factor_vars], factor) 

# Filter out any NA values
df_cleaned <- power_df %>% filter(!is.na(pl_theta_1_power))

# power ~ (b1*group + b2* period_type + b3*their interaction) + (b4 * animal ...) + error

# Run the model
pl_theta_1_model <-  lmer(pl_theta_1_power ~ group * period_type + (1|animal/period/event), 
                           data <- df_cleaned)

summary(pl_theta_1_model)



# Graph the predictions from the model
emmip(pl_theta_1_model, group ~ period_type, CIs = FALSE) + 
  labs(x = '', y = paste("Predicted BLA Theta 1 Power")) +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))


# get the residuals from the model
# plot them

residuals <- resid(pl_theta_1_model)

hist(residuals, main = "Histogram of Residuals")

plot(residuals ~ fitted(pl_theta_1_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")


# What do you do?

# 1. Check for outliers (clearly not relevant here -- it's not just a few points
# causing this problem).

# 2. Scale or transform the IV (not relevant here -- we have two categorical predictors)

# 3. Add predictors, for instance interaction terms if you didn't already, or non-linear 
# predictors, like polynomial predictors (x^2).  Also not relevant here.

# 4. Transform the DV so it better matches the distribution the model expects

# 5. Use a model that makes different assumptions about the distribution of the 
# underlying data


# Looking at the distribution of the underlying variables can give us a hint
hist(df_cleaned$pl_theta_1_power)

# Add the log of power to the data frame as a new column
df_cleaned$log_pl_theta_1_power <- log(df_cleaned$pl_theta_1_power)

# This log power is our new DV
pl_theta_1_model_log <-lmer(log_pl_theta_1_power ~ group * period_type + (1|animal/period/event), data <- df_cleaned)

summary(pl_theta_1_model_log)

emmip(pl_theta_1_model_log, group ~ period_type, CIs = FALSE) + 
  labs(x = '', y = paste("Predicted Ln PL Theta 1 Power")) +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))

# Check the residuals again

residuals <- resid(pl_theta_1_model_log)

hist(residuals, main = "Histogram of Residuals")

plot(residuals ~ fitted(_theta_1_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

#### FIRING RATE

rate_data <- read.csv('/Users/katie/likhtik/IG_INED_Safety_Recall/psth/psth_demo.csv', comment.char="#")


# a different command you can use to filter/subset data
rate_data <- subset(rate_data, rate_data$time < .3)
rate_data <- subset(rate_data, rate_data$time >= 0)
rate_data <- subset(rate_data, quality != '3')

# average over different time points within an event
rate_data <- rate_data %>%
  group_by(group, neuron_type, period_type, animal, unit, period, event) %>%
  summarise(rate = mean(rate, na.rm = TRUE), .groups = 'drop')


# Make sure every categorical variable is a factor
factor_vars <- c('animal', 'group', 'period_type', 'neuron_type', 'unit')
rate_data[factor_vars] <- lapply(rate_data[factor_vars], factor)

rate_model <- lmer(rate ~ group * period_type * neuron_type + (1|animal/unit/period), data = rate_data)

summary(rate_model)

residuals <- resid(rate_model)

hist(residuals, main = "Histogram of Residuals")

plot(residuals ~ fitted(rate_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")


hist(rate_data$rate)


### FIRING COUNT

# Load the CSV file
count_data <- 
  read.csv('/Users/katie/likhtik/IG_INED_Safety_Recall/spike_counts/spike_counts.csv', 
           comment.char="#")

count_data <- subset(count_data, count_data$time < .3)
count_data <- subset(count_data, count_data$time >= 0)
count_data <- subset(count_data, quality != '3')

count_data <- count_data %>%
  group_by(group, neuron_type, period_type, animal, unit, period, event) %>%
  summarise(count = sum(spike_counts, na.rm = TRUE), .groups = 'drop')


# Make sure every categorical variable is a factor
factor_vars <- c('animal', 'group', 'period_type', 'neuron_type', 'unit')
count_data[factor_vars] <- lapply(count_data[factor_vars], factor)


# What is a generalized mixed model?  It's like a regular mixed model, but 
# you take your regression predictors (here b1*group + b2*neuron_type, etc.)
# and you apply a function to them, here the natural log function, to make a 
# prediction. It's useful for data which have other kinds of distributions

count_model <- glmer(count ~ group * period_type * neuron_type + 
                                    (1|animal/unit/period), 
                                  family = poisson(link = "log"), 
                                  data = count_data)

# Model is singular! What do we do?
summary(count_model)

# Remove random effects with trivial variance

count_model_fewer_random_fx <- glmer(count ~ group * period_type * neuron_type + 
                                    (1|animal:unit) + (1|animal:unit:period), 
                                  family = poisson(link = "log"), 
                                  data = count_data)

# Poisson models aren't supposed to have normal residuals. How do we test them?

# Use the Dharma package to create simulated residuals that are transformed
# so they folow the uniform distribution and check their fit.

simulated_res = simulateResiduals(count_model_fewer_random_fx)
plot(simulated_res)
testOutliers(simulated_res)
testDispersion(simulated_res)

emm_results = emmeans(count_model_fewer_random_fx, specs = pairwise ~ group | period_type | neuron_type)
emm_means <- summary(emm_results, type = "response")  # This directly gives you the response scale values

count_data_plot <- emmip(emm_results, group ~ period_type | neuron_type, CIs = FALSE, type = "response") +
  labs(x="", y = "Predicted Count of Spikes per Event (0-.3s)") +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))

print(count_data_plot)

summary(count_model_fewer_random_fx)


# Can our model still be improved?

zip_model <- glmmTMB(count ~ group * period_type * neuron_type + (1|animal:unit) + (1|animal:unit:period),
                     ziformula = ~ 1,  # Zero-inflation part
                     family = poisson(link = "log"),
                     data = count_data)
summary(zip_model)

# How did you know about all this stuff, Katie?
# Honestly, ChatGPT.

simulated_res_zip = simulateResiduals(zip_model)
plot(simulated_res_zip)
testOutliers(simulated_res_zip)
testDispersion(simulated_res_zip)

# Which model is better?

AIC(count_model_fewer_random_fx)
AIC(zip_model)

# Let's choose our zero-inflated model!

emm_results = emmeans(zip_model, specs = pairwise ~ group | period_type | neuron_type)
emm_means <- summary(emm_results, type = "response")  # This directly gives you the response scale values

count_data_plot <- emmip(emm_results, group ~ period_type | neuron_type, CIs = FALSE, type = "response") +
  labs(x="", y = "Predicted Count of Spikes per Event (0-.3s)") +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))

print(count_data_plot)



# PERCENT FREEZING

just_freezing_csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/percent_freezing', 'freezing.csv', sep='/')
just_freezing_data <- read.csv(just_freezing_csv, comment.char="#")


factor_vars <- c('animal', 'group', 'period_type')
just_freezing_data[factor_vars] <- lapply(just_freezing_data[factor_vars], factor)


# Filter data for pretone and tone period types
just_freezing_data <- subset(just_freezing_data, period_type %in% c("pretone", "tone"))
just_freezing_data <- just_freezing_data[!is.na(just_freezing_data$percent_freezing), ]


periods_1 <- subset(just_freezing_data, period < 1)


# Basic mixed model
period_1_model <- lmer(percent_freezing ~ group*period_type + (1|animal), data=periods_1)


residuals <- resid(period_1_model)

hist(residuals, main = "Histogram of Residuals")

plot(residuals ~ fitted(period_1_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

# Not *horrendous* but can we do better?

hist(just_freezing_data$percent_freezing)

# Two things are true about the percent freezing data.  We disguise this because 
# it's multiplied by 100, but really it's an outcome variable that varies between
# 0 and 1.  There are specific regression types for this.  Also it has 
# disproportionate numbers of 1s. 

# make a new variable that varies between 0 and 1, non-inclusive (these types of 
# regression models often need the endpoints left off)

periods_1$proportion <- periods_1$percent_freezing/100
periods_1$percent_freezing_adj <- pmin(pmax(periods_1$proportion, 0.0001), 0.9999)


period_1_model_beta <- glmmTMB(percent_freezing_adj ~ group * period_type + (1 | animal),
                               data = periods_1,
                               family = list(family="beta", link="logit"))

residuals <- resid(period_1_model_beta)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

plot(residuals ~ fitted(period_1_model_beta ), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

hist(residuals, main = "Histogram of Residuals")

# Let's try a regression type that can model 1-inflated data.

model_gamlss <- gamlss(percent_freezing_adj ~ group * period_type + random(animal),
                       sigma.formula = ~ 1,  
                       nu.formula = ~ 1,  
                       family = BEOI(),  
                       data = periods_1)

summary(model_gamlss)

residuals <- resid(model_gamlss)
qqnorm(resid(model_gamlss))
qqline(resid(model_gamlss))

# The residuals are a little better; power is obviously better.


