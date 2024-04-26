library(lme4)
library(dplyr)
library(ggplot2)
library(emmeans)
library(rlang)

mrl_power_csv <- '/Users/katie/likhtik/IG_INED_Safety_Recall/power/mrl_power.csv'
mrl_power_data <- read.csv(mrl_power_csv, comment.char="#") 
read_metadata(mrl_power_csv)

all_lines <- readLines(mrl_power_data)

# Filter lines that start with the comment character
metadata_lines <- all_lines[grepl("^#", all_lines)]

# Print the metadata lines
cat(metadata_lines, sep = "\n")


factor_vars <- c('animal', 'group', 'period_type', 'neuron_type', 'unit')
mrl_power_data[factor_vars] <- lapply(mrl_power_data[factor_vars], factor)


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
    hpc_theta_2_power = mean(hpc_theta_2_power, na.rm = TRUE),
    bla_theta_2_power = mean(bla_theta_2_power, na.rm = TRUE),
    pl_theta_2_power  = mean(pl_theta_2_power, na.rm = TRUE),
    hpc_theta_2_mrl = mean(hpc_theta_2_mrl, na.rm = TRUE),
    bla_theta_2_mrl = mean(bla_theta_2_mrl, na.rm = TRUE),
    pl_theta_2_mrl  = mean(pl_theta_2_mrl, na.rm = TRUE),
    .groups = "drop"
  )

make_model <- function(dv, iv, data) {
  
  data$neuron_type <- factor(data$neuron_type,
                             levels = c("IN", "PN"))
  clean_data <- data[!is.na(data[[iv]]) & !is.na(data[['neuron_type']]), ]
  
  formula <- as.formula(paste(dv, "~", iv, "* period_type * neuron_type * group + (1|animal:unit)"))
  control = lmerControl(check.conv.grad=.makeCC("warning", tol=1e-4), optimizer="bobyqa")
  
  model <- lmer(formula, data = clean_data, control=control)
  print(summary(model))
  
  mean_iv <- mean(clean_data[[iv]], na.rm = TRUE)
  sd_iv <- sd(clean_data[[iv]], na.rm = TRUE)
  
  pred_data <- expand.grid(
    group = levels(clean_data$group),
    period_type = levels(clean_data$period_type),
    neuron_type = levels(clean_data$neuron_type),
    iv = c(mean_iv - sd_iv, mean_iv, mean_iv + sd_iv)
  )
  
  # Properly name the power variable in the prediction data
  names(pred_data)[names(pred_data) == "iv"] <- iv
  
  pred_data$predicted <- predict(model, newdata = pred_data, re.form = NA)
  
  plot <- graph_predictions(pred_data, iv, "predicted", iv, paste("Predicted", dv))
  
  return(list(data = pred_data, model = model, plot = plot))
}



variables <- c("hpc_theta_1_power", "bla_theta_1_power", "pl_theta_1_power", "pl_theta_2_power",
               "hpc_theta_1_mrl", "bla_theta_1_mrl", "pl_theta_1_mrl", "pl_theta_2_mrl")

combinations <- combn(variables, 2, simplify = FALSE)
filtered_combinations <- Filter(function(x) any(grepl("mrl", x)) && any(grepl("power", x)) , combinations)


results <- list()
for (i in seq_along(filtered_combinations)) {
  vars <- filtered_combinations[[i]]
  dv <- vars[2]
  iv <- vars[1]
  cat("Model and plot for DV:", dv, "and IV:", iv, "\n")
  results[[paste(dv, iv, sep = "_")]] <- make_model(dv, iv, data_with_mrl)
}

control = lmerControl(check.conv.grad=.makeCC("warning", tol=1e-4), optimizer="bobyqa")
data_with_mrl$neuron_type <- factor(data_with_mrl$neuron_type,
                                    levels = c("IN", "PN"))
data_with_mrl <- data_with_mrl[!is.na(data_with_mrl[['neuron_type']]), ]


### HPC MRL Theta 1 and MRL Power Theta 1 ###


clean_data <- data_with_mrl[!is.na(data_with_mrl[['hpc_theta_1_power']]), ]

modified_hpc_power_hpc_mrl <- lmer(hpc_theta_1_mrl ~ hpc_theta_1_power * period_type * neuron_type * group + (1|animal:unit), data = clean_data, control=control)
summary(modified_hpc_power_hpc_mrl)

pred_data = create_predictions_data(clean_data, modified_hpc_power_hpc_mrl, 'hpc_theta_1_power')
plot <- graph_predictions(pred_data, 'hpc_theta_1_power', "predicted", 'hpc_theta_1_power', paste("Predicted", 'hpc_theta_1_mrl'))


### PL MRL Theta 1 and PL Power Theta 1 ###


clean_data <- data_with_mrl[!is.na(data_with_mrl[['pl_theta_1_power']]), ]

modified_pl_power_pl_mrl <- lmer(pl_theta_1_mrl ~ pl_theta_1_power * period_type * neuron_type * group + (1|animal:unit), data = clean_data, control=control)
summary(modified_pl_power_pl_mrl)

pred_data = create_predictions_data(clean_data, modified_pl_power_pl_mrl, 'pl_theta_1_power')
plot <- graph_predictions(pred_data, 'pl_theta_1_power', "predicted", 'pl_theta_1_mrl', paste("Predicted", dv))



# Model and plot for DV: hpc_theta_1_mrl and IV: bla_theta_1_power 

iv = 'bla_theta_1_power'
dv = 'hpc_theta_1_mrl'
clean_data <- data_with_mrl[!is.na(data_with_mrl[[iv]]) & !is.na(data_with_mrl[[dv]]), ]
modified_bla_power_hpc_mrl <- lmer(hpc_theta_1_mrl ~ bla_theta_1_power * period_type * neuron_type * group + (1|animal:unit), data = clean_data, control=control)
summary(modified_bla_power_hpc_mrl)
pred_data = create_predictions_data(clean_data, modified_bla_power_hpc_mrl, iv)
plot <- graph_predictions(pred_data, iv, "predicted", iv, paste("Predicted", dv))


# Model and plot for DV: bla_theta_1_mrl and IV: bla_theta_1_power 

iv = 'bla_theta_1_power'
dv = 'bla_theta_1_mrl'
clean_data <- data_with_mrl[!is.na(data_with_mrl[[iv]]) & !is.na(data_with_mrl[[dv]]), ]
modified_bla_power_bla_mrl <- lmer(bla_theta_1_mrl ~ bla_theta_1_power * period_type * group + (1|animal/unit), data = clean_data, control=control)
summary(modified_bla_power_bla_mrl)
pred_data = create_predictions_data(clean_data, modified_bla_power_bla_mrl, iv)
plot <- graph_predictions(pred_data, iv, "predicted", iv, paste("Predicted", dv), num_vars=3)


# Model and plot for DV: pl_theta_1_mrl and IV: bla_theta_1_power 

iv = 'bla_theta_1_power'
dv = 'pl_theta_1_mrl'
clean_data <- data_with_mrl[!is.na(data_with_mrl[[iv]]) & !is.na(data_with_mrl[[dv]]), ]
modified_bla_power_pl_mrl <- lmer(pl_theta_1_mrl ~ bla_theta_1_power * period_type* neuron_type * group + (1|animal:unit), data = clean_data, control=control)
summary(modified_bla_power_pl_mrl)
pred_data = create_predictions_data(clean_data, modified_bla_power_pl_mrl, iv)
plot <- graph_predictions(pred_data, iv, "predicted", iv, paste("Predicted", dv))


# Model and plot for DV: hpc_theta_1_mrl and IV: pl_theta_1_power 

iv = 'pl_theta_1_power'
dv = 'hpc_theta_1_mrl'
clean_data <- data_with_mrl[!is.na(data_with_mrl[[iv]]) & !is.na(data_with_mrl[[dv]]), ]
modified_bla_power_pl_mrl <- lmer(hpc_theta_1_mrl ~ pl_theta_1_power*  period_type* neuron_type * group + (1|animal:unit), data = clean_data, control=control)
summary(modified_bla_power_pl_mrl)
pred_data = create_predictions_data(clean_data, modified_bla_power_pl_mrl, iv)
plot <- graph_predictions(pred_data, iv, "predicted", iv, paste("Predicted", dv), num_vars=4)


# Model and plot for DV: pl_theta_2_mrl and IV: pl_theta_1_power 

iv = 'pl_theta_1_power'
dv = 'pl_theta_2_mrl'
clean_data <- data_with_mrl[!is.na(data_with_mrl[[iv]]) & !is.na(data_with_mrl[[dv]]), ]
modified_pl_power_pl_theta1_mrl <- lmer(pl_theta_2_mrl ~ pl_theta_1_power *  period_type * neuron_type * group + (1|animal:unit), data = clean_data, control=control)
summary(modified_bla_power_pl_mrl)
pred_data = create_predictions_data(clean_data, modified_pl_power_pl_theta1_mrl, iv)
plot <- graph_predictions(pred_data, iv, "predicted", iv, paste("Predicted", dv), num_vars=4)


# Model and plot for DV: hpc_theta_1_mrl and IV: pl_theta_2_power 

iv = 'pl_theta_2_power'
dv = 'hpc_theta_1_mrl'
clean_data <- data_with_mrl[!is.na(data_with_mrl[[iv]]) & !is.na(data_with_mrl[[dv]]), ]
modified_pl_theta_2_power_and_hpc_theta_1_mrl <- lmer(hpc_theta_1_mrl ~ pl_theta_2_power *  period_type * neuron_type * group + (1|animal:unit), data = clean_data, control=control)
summary(modified_pl_theta_2_power_and_hpc_theta_1_mrl)
pred_data = create_predictions_data(clean_data, modified_pl_theta_2_power_and_hpc_theta_1_mrl, iv)
plot <- graph_predictions(pred_data, iv, "predicted", iv, paste("Predicted", dv), num_vars=4)


# Model and plot for DV: pl_theta_1_mrl and IV: pl_theta_2_power 

iv = 'pl_theta_2_power'
dv = 'pl_theta_1_mrl'
clean_data <- data_with_mrl[!is.na(data_with_mrl[[iv]]) & !is.na(data_with_mrl[[dv]]), ]
modified_pl_theta_2_power_and_pl_theta_1_mrl <- lmer(pl_theta_1_mrl ~ pl_theta_2_power *  period_type * neuron_type * group + (1|animal:unit), data = clean_data, control=control)
summary(modified_pl_theta_2_power_and_pl_theta_1_mrl)
pred_data = create_predictions_data(clean_data, modified_pl_theta_2_power_and_hpc_theta_1_mrl, iv)
plot <- graph_predictions(pred_data, iv, "predicted", iv, paste("Predicted", dv), num_vars=4)


# Model and plot for DV: pl_theta_2_mrl and IV: pl_theta_2_power 

iv = 'pl_theta_2_power'
dv = 'pl_theta_2_mrl'
clean_data <- data_with_mrl[!is.na(data_with_mrl[[iv]]) & !is.na(data_with_mrl[[dv]]), ]
modified_pl_theta_2_power_and_pl_theta_2_mrl <- lmer(pl_theta_2_mrl ~ pl_theta_2_power *  period_type * neuron_type * group + (1|animal:unit), data = clean_data, control=control)
summary(modified_pl_theta_2_power_and_pl_theta_2_mrl)
pred_data = create_predictions_data(clean_data, modified_pl_theta_2_power_and_pl_theta_2_mrl, iv)
plot <- graph_predictions(pred_data, iv, "predicted", iv, paste("Predicted", dv), num_vars=4)











