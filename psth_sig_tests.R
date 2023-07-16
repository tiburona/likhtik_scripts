
library(glmmTMB)
library(lme4)
library(lmerTest)  # for lmer p-values
library(readr)
library(sjPlot)
library(ggplot2)
library(broom)
library(emmeans)

prepare_df <- function(csv, data_type, observation_type, up_until = NA){
  df <- read_csv(csv)
  
  # Convert variables to factors
  factor_vars <- c('unit_num', 'animal', 'category', 'condition')
  df[factor_vars] <- lapply(df[factor_vars], factor)
  
  if (data_type == 'proportion' && observation_type == 'unit') {
    df$"proportion"[df$"proportion" == 0] <- df$"proportion"[df$"proportion" == 0] + 1e-6
  }
  
  if (!is.na(up_until)) {
    df <- subset(df, time_bin <= up_until)
  }
  
  return(df)
}


generate_formula <- function(analysis_type, dv, error) {
  if (analysis_type == 'omnibus') {
    formula_str = paste(dv, "~ condition * category * time_bin +", error)
  } else if (analysis_type == 'post-hoc-interaction') {
    formula_str =  paste(dv, "~ condition * category +", error)
  } else {
    formula_str = paste(dv, "~ condition +", error)
  }
  formula = as.formula(formula_str)
  return(formula)
}

perform_regression <- function(df, data_type, observation_type, analysis_type){
  if (observation_type == 'unit') {
    error = "(1|animal)"
    family = beta_family(link = "logit")
  } else {
    error = "(1|animal/unit_num)"
    family = binomial()
  }
  if (data_type == 'psth') {
    dv = "rate"
    formula = generate_formula(analysis_type, dv, error)
    model = lmer(formula=formula, data=df)
  } else {
    dv = "proportion"
    formula = generate_formula(analysis_type, dv, error)
    model = glmmTMB(formula=formula, data=df, family=family )
  }
  return(model)
}

plot_predictions <- function(model, df, data_type){
  # Create a data frame of predicted values
  df$prediction <- predict(model, newdata = df, type = "response")
  
  # Melt the data for plotting
  df_melted <- reshape2::melt(df, id.vars = c("time_bin", "prediction"), variable.name = "variable", value.name = "value")
  
  # Plot
  p <- ggplot(df_melted, aes(x = time_bin, y = value)) +
    geom_point() +
    geom_line(aes(y = prediction), color = "red") +
    facet_wrap(~variable, scales = "free_y") +
    labs(title = paste("Predictions for", data_type, "data"),
         x = "Time Bin",
         y = "Value",
         color = "Variable")
  
  print(p)
}




  
### Unit PSTH ###
csv = '/Users/katie/likhtik/data/psth_continuous_units.csv'
data = prepare_df(csv, 'psth', 'unit')
unit_psth_model = perform_regression(data, 'psth', 'unit', 'omnibus')

### Trials PSTH ###
csv = '/Users/katie/likhtik/data/psth_continuous_trials.csv'
data = prepare_df(csv, 'psth', 'trial')
trials_psth_model = perform_regression(data, 'psth', 'trial', 'omnibus')

### Unit Proportion Score ###
csv = '/Users/katie/likhtik/data/proportion_continuous_units.csv'
data = prepare_df(csv, 'proportion', 'unit')
unit_proportion_model = perform_regression(data, 'proportion', 'unit', 'omnibus')

### Trials Proportion Score ###
csv = '/Users/katie/likhtik/data/proportion_continuous_trials.csv'
data = prepare_df(csv, 'proportion', 'trial')
trials_proportion_model = perform_regression(data, 'proportion', 'trial', 'omnibus')

tab_model(unit_psth_model)
tab_model(trials_psth_model)
tab_model(unit_proportion_model)
tab_model(trials_proportion_model)

### Trials Proportion Score Fewer Bins### 

csv = '/Users/katie/likhtik/data/proportion_continuous_trials.csv'
data = prepare_df(csv, 'proportion', 'trial', up_until=55)
trials_proportion_up_until_55_model = perform_regression(data, 'proportion', 'trial', 'omnibus')

tab_model(trials_proportion_up_until_55_model)

plot_predictions <- function(model, fixed_factor1, fixed_factor2, fixed_factor3, continuous_values) {
  # Create a named list for the 'at' argument
  at_list <- setNames(list(continuous_values), fixed_factor3)
  
  # Construct formula from character strings
  formula <- reformulate(paste(fixed_factor2, fixed_factor3, sep=" * "), fixed_factor1)
  
  # Estimate marginal means
  emmeans_est <- emmeans(model, formula, at = at_list)
  
  # Generate interaction plot
  p <- emmip(emmeans_est, formula, CIs = TRUE, point = FALSE)
  
  print(p)
  return(p)
}

time_bin = seq(0, .7, by = 0.05)
category = c('IN', 'PN')
condition = c('control', 'stressed')
trials_psth_predictions_plot = emmip(trials_psth_model, category~time_bin | condition , 
               at=list(time_bin=time_bin, category=category, condition=condition), CIs=FALSE)


unit_psth_predictions_plot = plot_predictions(unit_psth_model, 'category', 'condition', 'time_bin', seq(0, .7, by = 0.1))
trials_psth_predictions_plot = plot_predictions(trials_psth_model, data, 'psth')
unit_proportion_predictions_plot = plot_predictions(unit_proportion_model, data, 'proportion')
trials_proportion_predictions_plot = plot_predictions(trials_proportion_model, data, 'proportion')




