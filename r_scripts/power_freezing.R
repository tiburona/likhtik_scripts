library(lme4)
library(nlme)
library(dplyr)
library(glmmTMB)
library(lmerTest)
library(sjPlot)
library(ggplot2)
library(emmeans)


csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/percent_freezing', 'psth_power_freezing.csv', sep='/')
data <- read.csv(csv, comment.char="#") 
read_metadata(csv)

factor_vars <- c('animal', 'group', 'period_type', 'unit', 'neuron_type')
data[factor_vars] <- lapply(data[factor_vars], factor)

power_data <- data %>%
  group_by(period, group, period_type, animal, event) %>%
  summarise(
    hpc_theta_1_power = mean(hpc_theta_1_power, na.rm = TRUE),
    bla_theta_1_power = mean(bla_theta_1_power, na.rm = TRUE),
    pl_theta_1_power  = mean(pl_theta_1_power, na.rm = TRUE),
    percent_freezing = mean(percent_freezing, na.rm = TRUE),
    hpc_theta_2_power = mean(hpc_theta_2_power, na.rm = TRUE),
    bla_theta_2_power = mean(bla_theta_2_power, na.rm = TRUE),
    pl_theta_2_power  = mean(pl_theta_2_power, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )


data_early_periods <- power_data %>%
  filter(period < 2)


create_predictions_data_no_nt <- function(data, model, continuous_predictor) {
  
  
  mean_iv <- mean(data[[continuous_predictor]], na.rm = TRUE)
  sd_iv <- sd(data[[continuous_predictor]], na.rm = TRUE)
  
  pred_data <- expand.grid(
    group = levels(clean_data$group),
    period_type = levels(clean_data$period_type),
    iv = c(mean_iv - sd_iv, mean_iv, mean_iv + sd_iv)
  )
  
  # Properly name the power variable in the prediction data
  names(pred_data)[names(pred_data) == "iv"] <- continuous_predictor
  
  pred_data$predicted <- predict(model, newdata = pred_data, re.form = NA)
  
  return(pred_data)
}

###BLA###

bla_freezing_power_model <- lmer(bla_theta_1_power ~ group*period_type*percent_freezing + (1|animal/period), 
                                 data = power_data)
summary(bla_freezing_power_model)

predictions <- create_predictions_data_no_nt(power_data, bla_freezing_power_model, 
                                       'percent_freezing')

p <- graph_predictions(data=predictions, x='percent_freezing', y='predicted', 
                       ylabel=paste('Predicted', 'BLA Theta 1 Power'), xlabel='Freezing', num_vars=3)


### PL ###

subset_data <- power_data[!is.na(power_data$percent_freezing) & !is.na(power_data$pl_theta_1_power), ]

pl_freezing_power_model <- lmer(pl_theta_1_power ~ group*period_type*percent_freezing + (1|animal/period), 
                                 data = subset_data)
summary(pl_freezing_power_model)

predictions <- create_predictions_data_no_nt(power_data, pl_freezing_power_model, 
                                             'percent_freezing')

p <- graph_predictions(data=predictions, x='percent_freezing', y='predicted', 
                       ylabel=paste('Predicted', 'PL Theta 1 Power'), xlabel='Percent Freezing', num_vars=3)


### HPC ###
subset_data <- power_data[!is.na(power_data$percent_freezing) & !is.na(power_data$hpc_theta_1_power), ]

hpc_freezing_power_model <- lmer(hpc_theta_1_power ~ group*period_type*percent_freezing + (1|animal/period), 
                                data = subset_data)
summary(hpc_freezing_power_model)

predictions <- create_predictions_data_no_nt(power_data, hpc_freezing_power_model, 
                                             'percent_freezing')

p <- graph_predictions(data=predictions, x='percent_freezing', y='predicted', 
                       ylabel=paste('Predicted', 'HPC Theta 1 Power'), xlabel='Percent Freezing', num_vars=3)



###BLA THETA 2###

subset_data <- power_data[!is.na(power_data$percent_freezing) & !is.na(power_data$bla_theta_2_power), ]

bla_freezing_power_model <- lmer(bla_theta_2_power ~ group*period_type*percent_freezing + (1|animal/period), 
                                 data = subset_data)
summary(bla_freezing_power_model)

predictions <- create_predictions_data_no_nt(power_data, bla_freezing_power_model, 
                                             'percent_freezing')

p <- graph_predictions(data=predictions, x='percent_freezing', y='predicted', 
                       ylabel=paste('Predicted', 'BLA Theta 2 Power'), xlabel='Freezing', num_vars=3)