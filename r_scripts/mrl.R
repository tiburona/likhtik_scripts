library(glmmTMB)
library(ggpattern)
library(lme4)
library(lmerTest)  # for lmer p-values
library(readr)
library(sjPlot)
library(ggplot2)
library(broom)
library(emmeans)
library(dplyr)
library(rlang)
library(readr)
library(betareg)


csv_dir = '/Users/katie/likhtik/IG_INED_Safety_Recall/mrl'
csv_name = 'mrl.csv'
csv_file = paste(csv_dir, csv_name, sep='/')
df <- read.csv(csv_file, comment.char="#") 
df <- subset(df, neuron_quality != '3')



prepare_df <- function(csv_file, frequency_band, brain_region, evoked=FALSE){
 
  df <- read.csv(csv_file, comment.char="#") 
  
  df <- subset(df, neuron_quality != '3')
  
  
  # Convert variables to factors
  factor_vars <- c('animal', 'group', 'period_type', 'neuron_type', 'unit')
  df[factor_vars] <- lapply(df[factor_vars], factor)
  
  
  
  # Convert frequency_band to a symbol for tidy evaluation
  mrl_column <- sym(paste0(brain_region, '_', frequency_band, "_mrl"))
  
  averaged_result <- df %>%
    group_by(animal, period_type, period, neuron_type, group, unit) %>%
    summarize(
      mean_mrl = mean(!!mrl_column, na.rm = TRUE)
    ) %>%
    ungroup()
  
  data_to_return <- averaged_result
  
  return(data_to_return) 
}



bootstrap_model <- function(fit, nsim=1000) {
  boot_results <- bootMer(fit, FUN=fixef, nsim)
  
  num_effects <- ncol(boot_results$t)
  
  # Initialize a matrix to store the confidence intervals:
  ci_matrix <- matrix(0, nrow=2, ncol=num_effects)
  
  # Loop through each effect to calculate confidence intervals:
  for (i in 1:num_effects) {
    ci_matrix[,i] <- quantile(boot_results$t[,i], c(0.025, 0.975))
  }
  
  # Convert matrix to data frame and add names for clarity:
  ci_df <- as.data.frame(t(ci_matrix))
  colnames(ci_df) <- c("2.5%", "97.5%")
  rownames(ci_df) <- names(fixef(fit))
  
  # Add an asterisk for effects where CI does not include 0:
  ci_df$significance <- ifelse(ci_df$`2.5%` > 0 | ci_df$`97.5%` < 0, "*", "")
  
  # Print the results:
  print(ci_df)
  
  # Return the data frame in case the user wants to save or manipulate it further:
  return(ci_df)
}



#### PL #####

### Theta 1 ###

df <- prepare_df(csv_file, 'theta_1', 'pl')
clean_df <- df %>% filter(!is.na(mean_mrl))

clean_df$sqrt_mrl <- clean_df$mean_mrl**.5

model_glmmTMB <- glmmTMB(sqrt_mrl ~ group * neuron_type * period_type + (1 | animal/unit), data = clean_df, family = beta_family(link = "logit"))

summary(model_glmmTMB)

simulationOutput <- simulateResiduals(fittedModel = model_glmmTMB, plot = TRUE)


emm_results = emmeans(model_glmmTMB, specs = pairwise ~ group | period_type | neuron_type)
emm_means <- summary(emm_results, type = "response")  # This directly gives you the response scale values



plot = emmip(emm_results, group ~ period_type | neuron_type, CIs = FALSE, type="response") + 
  labs(x="", y = paste("Predicted Square Root MRL PL Theta 1")) +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))


#### BLA #####

### Theta 1 ###


df <- prepare_df(csv_file, 'theta_1', 'bla')
clean_df <- df %>% filter(!is.na(mean_mrl))

clean_df$sqrt_mrl <- clean_df$mean_mrl**.5

model_glmmTMB <- glmmTMB(sqrt_mrl ~ group * neuron_type * period_type + (1 | animal/unit), data = clean_df, family = beta_family(link = "logit"))

summary(model_glmmTMB)

simulationOutput <- simulateResiduals(fittedModel = model_glmmTMB, plot = TRUE)


emm_results = emmeans(model_glmmTMB, specs = pairwise ~ group | period_type | neuron_type)
emm_means <- summary(emm_results, type = "response")  # This directly gives you the response scale values


plot = emmip(emm_results, group ~ period_type | neuron_type, CIs = FALSE, type="response") + 
  labs(x="", y = paste("Predicted Square Root MRL BLA Theta 1")) +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))

### HPC ###

# Theta 1 #


df <- prepare_df(csv_file, 'theta_1', 'hpc')
clean_df <- df %>% filter(!is.na(mean_mrl))

clean_df$sqrt_mrl <- clean_df$mean_mrl**.5

model_glmmTMB <- glmmTMB(sqrt_mrl ~ group * neuron_type * period_type + (1 | animal/unit), data = clean_df, family = beta_family(link = "logit"))

summary(model_glmmTMB)

simulationOutput <- simulateResiduals(fittedModel = model_glmmTMB, plot = TRUE)


emm_results = emmeans(model_glmmTMB, specs = pairwise ~ group | period_type | neuron_type)
emm_means <- summary(emm_results, type = "response")  # This directly gives you the response scale values


plot = emmip(emm_results, group ~ period_type | neuron_type, CIs = FALSE, type="response") + 
  labs(x="", y = paste("Predicted Square Root MRL BLA Theta 1")) +
  scale_color_manual(values = c("control" = "#6C4675", "defeat" = "#F2A354"))
