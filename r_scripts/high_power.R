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

csv_name = 'power_bla_(30, 50)_power_bla_(70, 120)_power_pl_(30, 50)_power_pl_(70, 120)_power_hpc_(30, 50)_power_hpc_(70, 120).csv'
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
    rename_with(~ gsub("_30_.50_", "gamma", .x, fixed = TRUE)) %>%
    rename_with(~ gsub("_70_.120_", "hgamma", .x, fixed = TRUE))  %>%
  
    group_by(across(all_of(group_vars))) %>%
    summarise(
      hpc_gamma_power = mean(hpc_gamma_power, na.rm = TRUE),
      bla_gamma_power = mean(bla_gamma_power, na.rm = TRUE),
      pl_gamma_power  = mean(pl_gamma_power, na.rm = TRUE),
      hpc_hgamma_power  = mean(hpc_hgamma_power, na.rm = TRUE), 
      bla_hgamma_power = mean(bla_hgamma_power, na.rm = TRUE),
      pl_hgamma_power  = mean(pl_hgamma_power, na.rm = TRUE),  
      .groups = "drop"
    )
}



grouped_df = group_and_summarize(power_df, c("period", "group", "period_type", "animal", "event"))

### Models ####

# BLA #

# gamma #

df_cleaned <- grouped_df %>% filter(!is.na(bla_gamma_power))
df_cleaned$log_bla_gamma_power <- log(df_cleaned$bla_gamma_power)
bla_gamma_model <-  lmer(log_bla_gamma_power ~ group * period_type + (1|animal/period), data <- df_cleaned)

summary(bla_gamma_model)

residuals <- resid(bla_gamma_model)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

plot(residuals ~ fitted(bla_gamma_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

hist(residuals, main = "Histogram of Residuals")

pdf("/Users/katie/likhtik/IG_INED_Safety_Recall/power/bla_gamma_model.pdf", width = 5, height = 7) 

emmip(bla_gamma_model, group ~ period_type, CIs = FALSE) + 
  labs(x = '', y = paste("Predicted Ln BLA gamma Power")) +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))

# Close the device
dev.off()

# Hgamma #

df_cleaned <- grouped_df %>% filter(!is.na(bla_hgamma_power))
df_cleaned$log_bla_hgamma_power <- log(df_cleaned$bla_hgamma_power)
bla_hgamma_model <-  lmer(log_bla_hgamma_power ~ group * period_type + (1|animal/period), data <- df_cleaned)

summary(bla_hgamma_model)

residuals <- resid(bla_hgamma_model)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

plot(residuals ~ fitted(bla_hgamma_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

hist(residuals, main = "Histogram of Residuals")

pdf("/Users/katie/likhtik/IG_INED_Safety_Recall/power/bla_gamma_model.pdf", width = 5, height = 7) 

emmip(bla_hgamma_model, group ~ period_type, CIs = FALSE) + 
  labs(x = '', y = paste("Predicted Ln BLA gamma Power")) +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))

# Close the device
dev.off()


# PL #

# gamma #

df_cleaned <- grouped_df %>% filter(!is.na(pl_gamma_power))
df_cleaned$log_pl_gamma_power <- log(df_cleaned$pl_gamma_power)
pl_gamma_model <-  lmer(log_pl_gamma_power ~ group * period_type + (1|animal/period), data <- df_cleaned)

summary(pl_gamma_model)

residuals <- resid(pl_gamma_model)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

plot(residuals ~ fitted(pl_gamma_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

hist(residuals, main = "Histogram of Residuals")

pdf("/Users/katie/likhtik/IG_INED_Safety_Recall/power/pl_gamma_model.pdf", width = 5, height = 7) 

emmip(pl_gamma_model, group ~ period_type, CIs = FALSE) + 
  labs(x = '', y = paste("Predicted Ln PL gamma Power")) +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))

# Close the device
dev.off()

# Hgamma #

df_cleaned <- grouped_df %>% filter(!is.na(pl_hgamma_power))
df_cleaned$log_pl_hgamma_power <- log(df_cleaned$pl_hgamma_power)
pl_hgamma_model <-  lmer(log_pl_hgamma_power ~ group * period_type + (1|animal/period), data <- df_cleaned)

summary(pl_hgamma_model)

residuals <- resid(pl_hgamma_model)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

plot(residuals ~ fitted(pl_hgamma_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

hist(residuals, main = "Histogram of Residuals")

pdf("/Users/katie/likhtik/IG_INED_Safety_Recall/power/pl_hgamma_model.pdf", width = 5, height = 7) 

emmip(pl_hgamma_model, group ~ period_type, CIs = FALSE) + 
  labs(x = '', y = paste("Predicted Ln PL Hgamma Power")) +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))

# Close the device
dev.off()


# HPC #

# gamma #

df_cleaned <- grouped_df %>% filter(!is.na(hpc_gamma_power))
df_cleaned$log_hpc_gamma_power <- log(df_cleaned$hpc_gamma_power)
hpc_gamma_model <-  lmer(log_hpc_gamma_power ~ group * period_type + (1|animal/period), data <- df_cleaned)

summary(hpc_gamma_model)

residuals <- resid(hpc_gamma_model)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

plot(residuals ~ fitted(hpc_gamma_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

hist(residuals, main = "Histogram of Residuals")

pdf("/Users/katie/likhtik/IG_INED_Safety_Recall/power/hpc_gamma_model.pdf", width = 5, height = 7) 

emmip(hpc_gamma_model, group ~ period_type, CIs = FALSE) + 
  labs(x = '', y = paste("Predicted Ln HPC gamma Power")) +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))

# Close the device
dev.off()


# Hgamma

df_cleaned <- grouped_df %>% filter(!is.na(hpc_hgamma_power))
df_cleaned$log_hpc_hgamma_power <- log(df_cleaned$hpc_hgamma_power)
hpc_hgamma_model <-  lmer(log_hpc_hgamma_power ~ group * period_type + (1|animal/period), data <- df_cleaned)

summary(hpc_hgamma_model)

residuals <- resid(hpc_hgamma_model)
qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

plot(residuals ~ fitted(hpc_hgamma_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

hist(residuals, main = "Histogram of Residuals")

pdf("/Users/katie/likhtik/IG_INED_Safety_Recall/power/hpc_gamma_model.pdf", width = 5, height = 7) 

emmip(hpc_hgamma_model, group ~ period_type, CIs = FALSE) + 
  labs(x = '', y = paste("Predicted Ln HPC Hgamma Power")) +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))

# Close the device
dev.off()
