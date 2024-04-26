library(lme4)
library(nlme)
library(dplyr)
library(glmmTMB)
library(lmerTest)
library(sjPlot)
library(ggplot2)
library(emmeans)
library(purrr)


power_csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/power', 'mrl_power.csv', sep='/')
power_df <- read.csv(power_csv, comment.char="#") 

factor_vars <- c('animal', 'group', 'period_type')
power_df[factor_vars] <- lapply(power_df[factor_vars], factor)
# 
# prepip_df <- power_df %>%
#   filter(time_bin < 15)
# 
# postpip_df <- power_df %>%
#   filter(time_bin > 14 & time_bin < 45)


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

data_frames <- list(
  all = power_df
  #prepip = prepip_df,
  #postpip = postpip_df
)

make_models_and_graphs <- function(brain_region, frequency_band, data_framees) {
  
  # Create the column name dynamically based on function inputs
  power_col_name <- paste(brain_region, frequency_band, "power", sep="_")
  
  
  # Prepare lists to store models and graphs
  models_list <- list()
  graphs_list <- list()
  
  # Iterate over the list of data frames with names
  for (df_name in names(data_frames)) {
    df <- data_frames[[df_name]]
    
    # Summarize data including 'time_bin'
    with_time_bin <- group_and_summarize(
      df, c("period", "group", "period_type", "animal", "event", "time_bin"))
    
    # Fit model using the dynamically specified power column
    model_with_bin <- lmer(
      as.formula(paste(power_col_name, "~ group * period_type + (1|animal/period/event)")), 
      data=with_time_bin)
    
    # Store model and print summary with title
    models_list[[paste("model_with_bin_", df_name, sep='')]] <- model_with_bin
    cat("\nSummary of Model with Time Bin (", df_name, "):\n", sep='')
    print(summary(model_with_bin))
    
    # Summarize data without 'time_bin'
    without_time_bin <- group_and_summarize(
      df, c("period", "group", "period_type", "animal", "event"))
    
    # Fit model for the data without 'time_bin'
    model_without_bin <- lmer(
      as.formula(paste(power_col_name, "~ group * period_type + (1|animal/period)")), 
      data=without_time_bin)
    
    # Store model and print summary with title
    models_list[[paste("model_without_bin_", df_name, sep='')]] <- model_without_bin
    cat("\nSummary of Model without Time Bin (", df_name, "):\n", sep='')
    print(summary(model_without_bin))
    
    # Generate graphs for each model
    plot_with_bin <- emmip(model_with_bin, group ~ period_type, CIs = FALSE) + 
      labs(y = paste("Predicted", toupper(brain_region), toupper(frequency_band), "Power")) +
      scale_color_manual(values = c("control" = "green", "defeat" = "orange"))
    
    plot_without_bin <- emmip(model_without_bin, group ~ period_type, CIs = FALSE) + 
      labs(y = paste("Predicted", toupper(brain_region), toupper(frequency_band), "Power")) +
      scale_color_manual(values = c("control" = "green", "defeat" = "orange"))
    
    # Store graphs with descriptive keys
    graphs_list[[paste("graph_with_bin_", df_name, sep='')]] <- plot_with_bin
    graphs_list[[paste("graph_without_bin_", df_name, sep='')]] <- plot_without_bin
  }
  
  # Return a list containing all models and graphs
  return(list("Models" = models_list, "Graphs" = graphs_list))
}


### Models ####


bla_theta_1_results <- make_models_and_graphs('bla', 'theta_1')

graphs <- bla_theta_1_results$Graphs

graphs$graph_with_bin_all

bla_theta_2_results <- make_models_and_graphs('bla', 'theta_2')

graphs <- bla_theta_2_results$Graphs


graphs$graph_with_bin_all


pl_theta_1_results <- make_models_and_graphs('pl', 'theta_1')

graphs <- pl_theta_1_results$Graphs

graphs$graph_with_bin_prepip
graphs$graph_with_bin_postpip
graphs$graph_with_bin_all

bla_theta_2_results <- make_models_and_graphs('bla', 'theta_2')

graphs <- bla_theta_2_results$Graphs

graphs$graph_with_bin_prepip
graphs$graph_with_bin_postpip
graphs$graph_with_bin_all

hpc_theta_1_results <- make_models_and_graphs('hpc', 'theta_1')

graphs <- hpc_theta_1_results$Graphs

graphs$graph_with_bin_all


hpc_theta_2_results <- make_models_and_graphs('hpc', 'theta_2')
graphs <- hpc_theta_2_results$Graphs

graphs$graph_with_bin_all



run_power_power_analysis <- function(data, region1, frequency_band1, region2, frequency_band2) {
  
  # Dynamically build the variable names based on region and frequency
  power_var1 <- paste(region1, frequency_band1, "power", sep = "_")
  power_var2 <- paste(region2, frequency_band2, "power", sep = "_")
  
  # Filter out rows where the power variable is NA
  clean_data <- data[!is.na(data[[power_var1]]) & !is.na(data[[power_var2]]), ]
  
  power_vars = c(power_var1, power_var2)
  
  # Define control parameters for the linear mixed-effects model
  control = lmerControl(check.conv.grad=.makeCC("warning", tol=1e-4), optimizer="bobyqa")
  
  # Build the formula string dynamically
  formula <- as.formula(paste(power_var2, " ~ group*period_type*", power_var1,
                              "+ (1|animal/period)", sep = ""))
  
  # Fit the model
  model <- lmer(formula, data = clean_data, control = control)
  
  # Display the summary of the model
  print(summary(model))
  
  # Calculate mean and standard deviation for the power variable
  mean_power <- mean(data[[power_var1]], na.rm = TRUE)
  sd_power <- sd(data[[power_var1]], na.rm = TRUE)
  
  pred_data <- expand.grid(
    group = levels(clean_data$group),
    period_type = levels(clean_data$period_type),
    power = c(mean_power - sd_power, mean_power, mean_power + sd_power)
  )
  
  # Properly name the power variable in the prediction data
  names(pred_data)[names(pred_data) == "power"] <- power_var1
  
  pred_data[[paste('predicted', power_var2, sep='_')]] <- predict(model, newdata = pred_data, re.form = NA)
  
  # Plot the predictions
  p <- graph_predictions(data=pred_data, 
                         x=power_var1, y=paste('predicted', power_var2, sep='_'), 
                         xlabel=paste(toupper(region1), frequency_band1, "power", sep = " "), 
                         ylabel=paste('Predicted', power_var2, sep=" "), num_vars = 3)
  
  # Return both the model and the plot
  return(list(model = model, plot = p))
}

grouped_df = group_and_summarize(power_df, c("period", "group", "period_type", "animal", "event"))
result <- run_power_power_analysis(grouped_df, 'bla', 'theta_1', 'pl', 'theta_1')


result$plot


grouped_df = group_and_summarize(power_df, c("period", "group", "period_type", "animal", "event"))
result <- run_power_power_analysis(grouped_df, 'hpc', 'theta_2', 'pl', 'theta_2')


result$plot