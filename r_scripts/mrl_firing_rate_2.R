library(lme4)
library(dplyr)
library(ggplot2)
library(emmeans)
library(rlang)
library(lmerTest)



control = lmerControl(check.conv.grad=.makeCC("warning", tol=1e-4), optimizer="bobyqa")
csv_dir = '/Users/katie/likhtik/IG_INED_Safety_Recall/mrl'

csv_name = 'psth_mrl.csv'

csv_file = paste(csv_dir, csv_name, sep='/')
mrl_rate_df <- read.csv(csv_file, comment.char="#") 


read_metadata(csv_file)

mrl_rate_df$neuron_type <- factor(mrl_rate_data$neuron_type,
                           levels = c("IN", "PN"))
mrl_rate_df <- mrl_rate_df[!is.na(mrl_rate_data[['neuron_type']]), ]


factor_vars <- c('animal', 'group', 'period_type', 'unit')
mrl_rate_df[factor_vars] <- lapply(mrl_rate_data[factor_vars], factor)

mrl_rate_data <- mrl_rate_df %>%
  filter(time_bin < 60) %>%
  group_by(period, group, period_type, neuron_type, unit, animal, event) %>%
  summarise(
    rate = mean(rate, na.rm = TRUE),
    hpc_theta_1_mrl = mean(hpc_theta_1_mrl, na.rm = TRUE),
    bla_theta_1_mrl = mean(bla_theta_1_mrl, na.rm = TRUE),
    pl_theta_1_mrl  = mean(pl_theta_1_mrl, na.rm = TRUE),
    pl_theta_2_mrl = mean(pl_theta_2_mrl, na.rm = TRUE),
    bla_theta_2_mrl = mean(bla_theta_2_mrl, na.rm = TRUE),
    hpc_theta_2_mrl = mean(hpc_theta_2_mrl, na.rm = TRUE),
    .groups = "drop"  # This line drops the grouping structure and returns a regular data frame
  )



make_mrl_rate_model <- function(iv, data) {
  clean_data <- mrl_rate_df[!is.na(data[[iv]]), ]
  
  formula <- as.formula(paste("rate ~", iv, " * period_type * neuron_type * group + (1|animal:unit:period)"))
  model <- lmer(formula, data = clean_data)
  print(summary(model))
  
  mean_iv <- mean(clean_data[[iv]], na.rm = TRUE)
  sd_iv <- sd(clean_data[[iv]], na.rm = TRUE)
  
  predictions_data <- create_predictions_data(clean_data, model, iv)
  plot <- graph_predictions(predictions_data, iv, "predicted", iv, paste("Predicted", 'rate'))
  
  return(list(data = predictions_data, model = model, plot = plot))
}

variables <- c("hpc_theta_1_mrl", "bla_theta_1_mrl", "pl_theta_1_mrl")

results <- list()
for (i in seq_along(variables)) {
  iv <- variables[i]
  cat("Model and plot for IV:", iv, "\n")
  results[[iv]] <- make_mrl_rate_model(iv, mrl_rate_data)
}

summary(results[['hpc_theta_1_mrl']]$model)
results[['hpc_theta_1_mrl']]$plot

summary(results[['pl_theta_1_mrl']]$model)
results[['pl_theta_1_mrl']]$plot

summary(results[['bla_theta_1_mrl']]$model)
results[['bla_theta_1_mrl']]$plot


### HPC MRL Theta 1 and Rate ###


clean_data <- mrl_rate_data[!is.na(mrl_rate_data[['hpc_theta_1_mrl']]), ]

modified_hpc_theta_1_mrl_rate <- lmer(rate ~ hpc_theta_1_mrl * period_type * group + (1|animal) + (1|animal:unit:period), data = clean_data, control=control)
summary(modified_hpc_theta_1_mrl_rate)

pred_data = create_predictions_data(clean_data, modified_hpc_theta_1_mrl_rate, 'hpc_theta_1_mrl')
plot <- graph_predictions(pred_data, 'hpc_theta_1_mrl', "predicted", 'hpc_theta_1_mrl', paste("Predicted", 'Rate'), num_vars=3)


### BLA MRL Theta 1 and Rate ###


clean_data <- mrl_rate_data[!is.na(mrl_rate_data[['bla_theta_1_mrl']]), ]
modified_bla_theta_1_mrl_rate <- lmer(rate ~ bla_theta_1_mrl * neuron_type* period_type * group + (1|animal) + (1|animal:unit:period), data = clean_data, control=control)
summary(modified_bla_theta_1_mrl_rate)
pred_data = create_predictions_data(clean_data, modified_bla_theta_1_mrl_rate, 'bla_theta_1_mrl')
plot <- graph_predictions(pred_data, 'bla_theta_1_mrl', "predicted", 'bla_theta_1_mrl', paste("Predicted", 'Rate'), num_vars=4)


### PL MRL Theta 1 and Rate ###


clean_data <- mrl_rate_data[!is.na(mrl_rate_data[['pl_theta_1_mrl']]), ]
modified_pl_theta_1_mrl_rate <- lmer(rate ~ pl_theta_1_mrl * neuron_type* period_type * group + (1|animal) + (1|animal:unit:period), data = clean_data, control=control)
summary(modified_pl_theta_1_mrl_rate)
pred_data = create_predictions_data(clean_data, modified_pl_theta_1_mrl_rate, 'pl_theta_1_mrl')
plot <- graph_predictions(pred_data, 'pl_theta_1_mrl', "predicted", 'pl_theta_1_mrl', paste("Predicted", 'Rate'), num_vars=4)


### PL MRL Theta 2 and Rate ###


clean_data <- mrl_rate_data[!is.na(mrl_rate_data[['pl_theta_2_mrl']]), ]
modified_pl_theta_2_mrl_rate <- lmer(rate ~ pl_theta_2_mrl * neuron_type* period_type * group + (1|animal) + (1|animal:unit:period), data = clean_data, control=control)
summary(modified_pl_theta_2_mrl_rate)
pred_data = create_predictions_data(clean_data, modified_pl_theta_2_mrl_rate, 'pl_theta_2_mrl')
plot <- graph_predictions(pred_data, 'pl_theta_2_mrl', "predicted", 'pl_theta_2_mrl', paste("Predicted", 'Rate'), num_vars=4)



### BLA MRL Theta 2 and Rate ###


clean_data <- mrl_rate_data[!is.na(mrl_rate_data[['bla_theta_2_mrl']]), ]
modified_bla_theta_2_mrl_rate <- lmer(rate ~ bla_theta_2_mrl * neuron_type* period_type * group + (1|animal) + (1|animal:unit:period), data = clean_data, control=control)
summary(modified_bla_theta_2_mrl_rate)
pred_data = create_predictions_data(clean_data, modified_bla_theta_2_mrl_rate, 'bla_theta_2_mrl')
plot <- graph_predictions(pred_data, 'bla_theta_2_mrl', "predicted", 'bla_theta_2_mrl', paste("Predicted", 'Rate'), num_vars=4)



### HPC MRL Theta 2 and Rate ###


clean_data <- mrl_rate_data[!is.na(mrl_rate_data[['hpc_theta_2_mrl']]), ]
modified_hpc_theta_2_mrl_rate <- lmer(rate ~ hpc_theta_2_mrl * neuron_type* period_type * group + (1|animal/unit/period), data = clean_data, control=control)
summary(modified_hpc_theta_2_mrl_rate)
pred_data = create_predictions_data(clean_data, modified_hpc_theta_2_mrl_rate, 'hpc_theta_2_mrl')
plot <- graph_predictions(pred_data, 'hpc_theta_2_mrl', "predicted", 'hpc_theta_2_mrl', paste("Predicted", 'Rate'), num_vars=4)
