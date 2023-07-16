library(lme4)
library(readr)
library(glmmTMB)
library(sjPlot)
library(ggplot2)

df <- read_csv('/Users/katie/likhtik/data/proportion_score_continuous_trials.csv')

# Convert variables to factors
factor_vars <- c('unit_num', 'animal', 'category', 'condition')
df[factor_vars] <- lapply(df[factor_vars], factor)

df <- subset(df, time_bin <= 54)

formula <- proportion ~ category * condition * time_bin + 
  (1|animal/unit_num/trial)

# Fit the logistic regression model
logistic_model <- glmmTMB(formula, data = df, family = binomial())
summary(logistic_model)
tab_model(logistic_model)

unit_df <- read_csv('/Users/katie/likhtik/data/proportion_score_continuous_units.csv')

# Convert variables to factors
factor_vars <- c('unit_num', 'animal', 'category', 'condition')
unit_df[factor_vars] <- lapply(unit_df[factor_vars], factor)

# Add small constant to 0s in the data column
unit_df$"proportion"[unit_df$"proportion" == 0] <- unit_df$"proportion"[unit_df$"proportion" == 0] + 1e-6

beta_formula <- proportion ~ category * condition * time_bin + 
  (1|animal)

beta_model <- glmmTMB(
  formula = beta_formula,
  family = beta_family(link = "logit"),
  data = unit_df
)

summary(beta_model)
tab_model(beta_model)


# Subset the data frame to only include rows where 'time_bin' is 59 or less
df <- subset(df, time_bin <= 59)
df$time_split2 <- ifelse(df$time_bin <= 29, "early", "late")
df$time_split2 <- factor(df$time_split2)


unit_df <- subset(unit_df, time_bin <= 59)
unit_df$time_split2 <- ifelse(unit_df$time_bin <= 29, "early", "late")
unit_df$time_split2 <- factor(unit_df$time_split2)

two_split_beta_formula  <- proportion ~ category * condition * time_split2 + 
  (1|animal/unit_num/time_bin)

two_split_beta_model <- glmmTMB(
  formula = two_split_beta_formula,
  family = beta_family(link = "logit"),
  data = unit_df
)

unit_df$time_split3 <- ifelse(unit_df$time_bin < 5, "pip",
                         ifelse(unit_df$time_bin < 30, "early",
                                "late"))
unit_df$time_split3 <- factor(unit_df$time_split3)


three_split_beta_formula  <- proportion ~ category * condition * time_split3 + 
  (1|animal/unit_num/time_bin)

three_split_beta_model <- glmmTMB(
  formula = three_split_beta_formula,
  family = beta_family(link = "logit"),
  data = unit_df
)


tab_model(beta_model)

unit_graph <- ggplot(data = unit_df,
                     mapping = aes(x = factor(category, levels = c("IN", "PN")),
                                   y = proportion,
                                   fill = factor(time_split3, levels = c("pip",
                                                                    "early",
                                                                    "late")))) +
  geom_boxplot() +
  facet_wrap(~ condition, ncol = 1)

unit_graph_with_different_positions <- ggplot(data = unit_df,
                                              mapping = aes(x = factor(category, levels = c("IN", "PN")),
                                                            y = proportion,
                                                            fill = factor(condition, levels = c("control",
                                                                                                "stressed")))) +
  geom_boxplot(outlier.shape = NA) + # don't draw outliers
  facet_wrap(~ time_split3, ncol = 1)


pip_df = subset(unit_df, unit_df$time_split3 == 'pip')
early_df = subset(unit_df, unit_df$time_split3 == 'early')
late_df = subset(unit_df, unit_df$time_split3 == 'late')


control_PN_df = subset(unit_df, unit_df$condition == 'control' && unit_df$category == 'PN')
control_IN_df = subset(unit_df, unit_df$condition == 'control' && unit_df$category == 'IN')
stressed_PN_df = subset(unit_df, unit_df$condition == 'stressed' && unit_df$category == 'PN')
stressed_IN_df = subset(unit_df, unit_df$condition == 'stressed' && unit_df$category == 'IN')


control_PN_df_pip_early = (subset(control_PN_df, control_PN_df$time_split3 != 'late'))
control_PN_df_early_late = (subset(control_PN_df, control_PN_df$time_split3 != 'pip'))
control_PN_df_pip_late = (subset(control_PN_df, control_PN_df$time_split3 != 'early'))

control_PN_df_pip_early = (subset(control_PN_df, control_PN_df$time_split3 != 'late'))
control_PN_df_early_late = (subset(control_PN_df, control_PN_df$time_split3 != 'pip'))
control_PN_df_pip_late = (subset(control_PN_df, control_PN_df$time_split3 != 'early'))

control_PN_df_pip_early = (subset(control_PN_df, control_PN_df$time_split3 != 'late'))
control_PN_df_early_late = (subset(control_PN_df, control_PN_df$time_split3 != 'pip'))
control_PN_df_pip_late = (subset(control_PN_df, control_PN_df$time_split3 != 'early'))




subset_formula  <- proportion ~ category * condition * time_bin + 
  (1|animal/unit_num)

pip_beta_model <- glmmTMB(
  formula = subset_formula,
  family = beta_family(link = "logit"),
  data = pip_df
)
summary(pip_beta_model)


early_beta_model <- glmmTMB(
  formula = subset_formula,
  family = beta_family(link = "logit"),
  data = early_df
)

summary(early_beta_model)

late_beta_model <- glmmTMB(
  formula = subset_formula,
  family = beta_family(link = "logit"),
  data = late_df
) 

summary(late_beta_model)




dev.off()



# Load the necessary libraries
library(boot)
library(dplyr)

# Define a function to calculate the mean proportion for each combination
calc_mean <- function(data, indices) {
  data <- data[indices,]
  return(mean(data$proportion))
}

# Create a data frame to store the bootstrapped standard errors
df_boot <- unit_df %>%
  group_by(category, condition, time_split3) %>%
  do({
    group_data <- .
    boot_result <- boot(group_data, calc_mean, R = 1000)
    data.frame(stderr = sd(boot_result$t))
  })

# Combine the means and bootstrapped standard errors into a single data frame
df_summary <- unit_df %>%
  group_by(category, condition, time_split3) %>%
  summarise(mean = mean(proportion), .groups = "drop") %>%
  bind_cols(df_boot)

time_split_2_unit_graph <- ggplot(data = unit_df,
                     mapping = aes(x = factor(condition, levels = c("control", "stressed")),
                                   y = proportion,
                                   fill = factor(time_split2, levels = c(
                                                                        "early",
                                                                         "late")))) +
  geom_boxplot() +
  facet_wrap(~ category, ncol = 1)

# Load required package
library(dplyr)

# Create a new column "averaged_time_bin" by taking floor division of 'time_bin' by 5
unit_df$averaged_time_bin <- floor(unit_df$time_bin / 5)

# Calculate mean proportion score within each group of 'animal', 'unit_num', 'averaged_time_bin', 'condition', 'category'
avg_df <- unit_df %>% 
  group_by(animal, unit_num, averaged_time_bin, condition, category) %>% 
  summarise(mean_proportion = mean(proportion), .groups = "drop")

# Convert 'averaged_time_bin', 'animal', and 'unit_num' to factors
avg_df$averaged_time_bin <- factor(avg_df$averaged_time_bin)
avg_df$animal <- factor(avg_df$animal)
avg_df$unit_num <- factor(avg_df$unit_num)

# Now you can fit your model, looping over each averaged_time_bin
df_list <- split(avg_df, avg_df$averaged_time_bin)

fit_beta_model <- function(df) {
  beta_formula <- mean_proportion ~ category * condition + (1|animal/unit_num)
  
  beta_model <- glmmTMB(
    formula = beta_formula,
    family = beta_family(link = "logit"),
    data = df
  )
  
  return(beta_model)
}

models_list <- lapply(df_list, fit_beta_model)
models_summaries <- lapply(models_list, summary)

psth_df <- read_csv('/Users/katie/likhtik/data/psth_score_continuous_trials.csv')


two_split_beta_formula  <- proportion ~ category * condition * time_split2 + 
  (1|animal/unit_num/time_bin)

psth_df <- read_csv('/Users/katie/likhtik/data/psth_score_continuous_trials.csv')


