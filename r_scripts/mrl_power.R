library(lme4)
library(dplyr)
library(ggplot2)
library(emmeans)
library(rlang)

mrl_power_csv <- '/Users/katie/likhtik/data/lfp/percent_freezing/spike_power_mrl.csv'
mrl_power_data <- read.csv(mrl_power_csv, comment.char="#") 

data_with_mrl <- mrl_power_data %>%
  filter(time_bin < 30) %>%
  group_by(period, group, period_type, neuron_type, animal, unit) %>%
  summarise(
    hpc_theta_1_power = mean(hpc_theta_1_power, na.rm = TRUE),
    bla_theta_1_power = mean(bla_theta_1_power, na.rm = TRUE),
    pl_theta_1_power  = mean(pl_theta_1_power, na.rm = TRUE),
    hpc_theta_1_mrl = mean(hpc_theta_1_mrl, na.rm = TRUE),
    bla_theta_1_mrl = mean(bla_theta_1_mrl, na.rm = TRUE),
    pl_theta_1_mrl  = mean(pl_theta_1_mrl, na.rm = TRUE),
    .groups = "drop"
  )

make_model <- function(dv, iv, data) {
  formula <- as.formula(paste(dv, "~", iv, "* period_type * neuron_type * group + (1|animal/unit)"))
  model <- lmer(formula, data = data)
  print(summary(model))
  
  mean_iv <- mean(data[[iv]], na.rm = TRUE)
  sd_iv <- sd(data[[iv]], na.rm = TRUE)
  
  predictions_data <- create_predictions_data(data, model, iv)
  plot <- graph_predictions(predictions_data, iv, "predicted", iv, paste("Predicted", dv))
  
  return(list(data = predictions_data, model = model, plot = plot))
}



variables <- c("hpc_theta_1_power", "bla_theta_1_power", "pl_theta_1_power", 
               "hpc_theta_1_mrl", "bla_theta_1_mrl", "pl_theta_1_mrl")

combinations <- combn(variables, 2, simplify = FALSE)
filtered_combinations <- Filter(function(x) any(grepl("power", x)) && any(grepl("mrl", x)), combinations)



results <- list()
for (i in seq_along(filtered_combinations)) {
  vars <- filtered_combinations[[i]]
  dv <- vars[1]
  iv <- vars[2]
  cat("Model and plot for DV:", dv, "and IV:", iv, "\n")
  results[[paste(dv, iv, sep = "_")]] <- make_model(dv, iv, data_with_mrl)
}

results[["hpc_theta_1_power_bla_theta_1_power"]][["plot"]]
results[["bla_theta_1_power_pl_theta_1_power"]][["plot"]]
results[["pl_theta_1_power_hpc_theta_1_mrl"]][["plot"]]
results[["pl_theta_1_power_pl_theta_1_mrl"]][["plot"]]
results[["hpc_theta_1_mrl_bla_theta_1_mrl"]][["plot"]]
results[["hpc_theta_1_mrl_pl_theta_1_mrl"]][["plot"]]


predictions_data = results[["pl_theta_1_power_pl_theta_1_mrl"]][["data"]]

summary(predictions_data[predictions_data$group == "control", ])
summary(predictions_data[predictions_data$group == "stressed", ])


summary(data_with_mrl[predictions_data$group == "control", ])
summary(predictions_data[predictions_data$group == "stressed", ])









