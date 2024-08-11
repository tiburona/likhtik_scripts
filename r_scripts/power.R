library(lme4)
library(nlme)
library(dplyr)
library(glmmTMB)
library(lmerTest)
library(sjPlot)
library(ggplot2)
library(emmeans)
library(purrr)
library(MASS)
library(forecast)

csv_name = 'power.csv'
power_csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/power', csv_name, sep='/')
power_df <- read.csv(power_csv, comment.char="#") 

factor_vars <- c('animal', 'group', 'period_type')
power_df[factor_vars] <- lapply(power_df[factor_vars], factor)


group_and_summarize <- function(df, group_vars) {
  # Ensure that group_vars is a vector of characters
  if (!is.character(group_vars)) {
    stop("group_vars must be a character vector of column names")
  }
  
  # Dynamically create the grouping and summarization
  df %>%
    group_by(across(all_of(group_vars))) %>%
    summarise(
      hpc_theta_1_power = mean(hpc_theta_1_power, na.rm = TRUE),
      bla_theta_1_power = mean(bla_theta_1_power, na.rm = TRUE),
      pl_theta_1_power  = mean(pl_theta_1_power, na.rm = TRUE),
      hpc_theta_2_power  = mean(hpc_theta_2_power, na.rm = TRUE), 
      bla_theta_2_power = mean(bla_theta_2_power, na.rm = TRUE),
      pl_theta_2_power  = mean(pl_theta_2_power, na.rm = TRUE),  
      .groups = "drop"
    )
}



grouped_df = group_and_summarize(power_df, c("period", "group", "period_type", "animal", "event"))

### Models ####

# BLA #

# THETA 1 #

df_cleaned <- grouped_df %>% filter(!is.na(bla_theta_1_power))
df_cleaned$log_bla_theta_1_power <- log(df_cleaned$bla_theta_1_power)
bla_theta_1_model <-  lmer(log_bla_theta_1_power ~ group * period_type + (1|animal/period), data <- df_cleaned)

summary(bla_theta_1_model)

residuals <- resid(bla_theta_1_model)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

plot(residuals ~ fitted(bla_theta_1_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

hist(residuals, main = "Histogram of Residuals")

pdf("/Users/katie/likhtik/IG_INED_Safety_Recall/power/bla_theta_1_model.pdf", width = 5, height = 7) 

emmip(bla_theta_1_model, group ~ period_type, CIs = FALSE) + 
  labs(x = '', y = paste("Predicted Ln BLA Theta 1 Power")) +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))

# Close the device
dev.off()

# THETA 2 #

df_cleaned <- grouped_df %>% filter(!is.na(bla_theta_2_power))
df_cleaned$log_bla_theta_2_power <- log(df_cleaned$bla_theta_2_power)
bla_theta_2_model <-  lmer(log_bla_theta_2_power ~ group * period_type + (1|animal/period), data <- df_cleaned)

summary(bla_theta_2_model)

residuals <- resid(bla_theta_2_model)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

plot(residuals ~ fitted(bla_theta_2_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

hist(residuals, main = "Histogram of Residuals")

pdf("/Users/katie/likhtik/IG_INED_Safety_Recall/power/bla_theta_1_model.pdf", width = 5, height = 7) 

emmip(bla_theta_2_model, group ~ period_type, CIs = FALSE) + 
  labs(x = '', y = paste("Predicted Ln BLA Theta 1 Power")) +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))

# Close the device
dev.off()


# PL #

# THETA 1 #

df_cleaned <- grouped_df %>% filter(!is.na(pl_theta_1_power))
df_cleaned$log_pl_theta_1_power <- log(df_cleaned$pl_theta_1_power)
pl_theta_1_model <-  lmer(log_pl_theta_1_power ~ group * period_type + (1|animal/period), data <- df_cleaned)

summary(pl_theta_1_model)

residuals <- resid(pl_theta_1_model)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

plot(residuals ~ fitted(pl_theta_1_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

hist(residuals, main = "Histogram of Residuals")

pdf("/Users/katie/likhtik/IG_INED_Safety_Recall/power/pl_theta_1_model.pdf", width = 5, height = 7) 

emmip(pl_theta_1_model, group ~ period_type, CIs = FALSE) + 
  labs(x = '', y = paste("Predicted Ln PL Theta 1 Power")) +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))

# Close the device
dev.off()

# THETA 2 #

df_cleaned <- grouped_df %>% filter(!is.na(pl_theta_2_power))
df_cleaned$log_pl_theta_2_power <- log(df_cleaned$pl_theta_2_power)
pl_theta_2_model <-  lmer(log_pl_theta_2_power ~ group * period_type + (1|animal/period), data <- df_cleaned)

summary(pl_theta_2_model)

residuals <- resid(pl_theta_2_model)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

plot(residuals ~ fitted(pl_theta_2_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

hist(residuals, main = "Histogram of Residuals")

pdf("/Users/katie/likhtik/IG_INED_Safety_Recall/power/pl_theta_2_model.pdf", width = 5, height = 7) 

emmip(pl_theta_2_model, group ~ period_type, CIs = FALSE) + 
  labs(x = '', y = paste("Predicted Ln PL Theta 2 Power")) +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))

# Close the device
dev.off()


# HPC #

# THETA 1 #

df_cleaned <- grouped_df %>% filter(!is.na(hpc_theta_1_power))
df_cleaned$log_hpc_theta_1_power <- log(df_cleaned$hpc_theta_1_power)
hpc_theta_1_model <-  lmer(log_hpc_theta_1_power ~ group * period_type + (1|animal/period), data <- df_cleaned)

summary(hpc_theta_1_model)

residuals <- resid(hpc_theta_1_model)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

plot(residuals ~ fitted(hpc_theta_1_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

hist(residuals, main = "Histogram of Residuals")

pdf("/Users/katie/likhtik/IG_INED_Safety_Recall/power/hpc_theta_1_model.pdf", width = 5, height = 7) 

emmip(hpc_theta_1_model, group ~ period_type, CIs = FALSE) + 
  labs(x = '', y = paste("Predicted Ln HPC Theta 1 Power")) +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))

# Close the device
dev.off()


# THETA 2

df_cleaned <- grouped_df %>% filter(!is.na(hpc_theta_2_power))
df_cleaned$log_hpc_theta_2_power <- log(df_cleaned$hpc_theta_2_power)
hpc_theta_2_model <-  lmer(log_hpc_theta_2_power ~ group * period_type + (1|animal/period), data <- df_cleaned)

summary(hpc_theta_2_model)

residuals <- resid(hpc_theta_2_model)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

plot(residuals ~ fitted(hpc_theta_2_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

hist(residuals, main = "Histogram of Residuals")

pdf("/Users/katie/likhtik/IG_INED_Safety_Recall/power/hpc_theta_1_model.pdf", width = 5, height = 7) 

emmip(hpc_theta_2_model, group ~ period_type, CIs = FALSE) + 
  labs(x = '', y = paste("Predicted Ln HPC Theta 2 Power")) +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))

# Close the device
dev.off()
