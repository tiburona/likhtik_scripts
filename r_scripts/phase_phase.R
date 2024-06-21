library(dplyr)
library(tidyr)
library(ggplot2)
library(lme4)
library(stringr)

csv = paste('/Users/katie/likhtik/CH_EXT/phase_phase_mrl', 'phase_phase.csv', sep='/')
data <- read.csv(csv, comment.char="#") 
read_metadata(csv)

il_bf_csv = paste('/Users/katie/likhtik/CH_EXT/phase_phase_mrl', 'phase_phase_il_bf.csv', sep='/')
il_bf_data <- read.csv(il_bf_csv, comment.char="#") 

library(dplyr)
library(tidyr)
library(ggplot2)
library(lme4)
library(stringr)

process_data_and_plot <- function(data, region_set, frequency_band) {
  regions <- unlist(strsplit(region_set, "_"))
  region1 <- toupper(regions[1])
  region2 <- toupper(regions[2])
  frequency_band_formatted <- gsub("_", " ", tools::toTitleCase(frequency_band))  # Capitalize and replace underscores
  variable_name <- paste(region_set, frequency_band, "phase_phase_mrl", sep = "_")
  
  # Preprocess the data
  clean_data <- data %>%
    filter(!is.na(.data[[variable_name]])) %>%
    mutate(
      period = as.integer(str_extract(phase_relationship_calculator, "\\d+$")),
      time = case_when(
        period %in% c(0, 1) ~ "early",
        period %in% c(18, 19) ~ "late",
        TRUE ~ NA_character_
      ),
      time = as.factor(time)
    ) %>%
    filter(period %in% c(0, 1, 18, 19))
  
  # Create result data frame
  result_data <- clean_data %>%
    filter(period_type %in% c("pretone", "tone")) %>%
    pivot_wider(
      names_from = period_type,
      values_from = !!sym(variable_name)
    ) %>%
    group_by(animal, time) %>%
    mutate(difference = tone - pretone) %>%
    distinct(animal, time, .keep_all = TRUE) %>%
    select(animal, time, difference)
  
  # Statistical modeling
  model_formula <- as.formula(paste(variable_name, "~ time * period_type + (1|animal)"))
  model <- lmer(model_formula, data = clean_data)
  model_summary <- summary(model)
  
  # Calculate means and standard errors for plotting
  graph_data <- result_data %>%
    group_by(time) %>%
    summarise(
      mean = mean(difference, na.rm = TRUE),
      sd = sd(difference, na.rm = TRUE),
      n = n(),
      se = sd / sqrt(n)
    )
  
  # Plotting
  plot <- ggplot(graph_data, aes(x = time, y = mean, fill = time)) +
    geom_bar(stat = "identity", position = position_dodge(), width = 0.7) +
    geom_errorbar(
      aes(ymin = mean - se, ymax = mean + se),
      width = 0.2,
      position = position_dodge(0.7)
    ) +
    labs(
      title = sprintf("Evoked Phase-Phase %s MRL Between %s and %s", frequency_band_formatted, region1, region2),
      x = NULL,  # No x-axis label
      y = "MRL"
    ) +
    theme_minimal() +
    scale_fill_manual(values = c("early" = "#513685", "late" = "#a158c7")) +
    scale_x_discrete(labels = c("early" = "Periods 1 & 2", "late" = "Periods 19 & 20"))
  
  # Return list containing the model summary and the plot
  list(model_summary = model_summary, plot = plot)
}




bla_il_result <- process_data_and_plot(data,  "bla_il", "theta_1")
print(bla_il_result$model_summary)
print(bla_il_result$plot)


il_bf_result <- process_data_and_plot(il_bf_data, 'il_bf', "theta_1")
print(il_bf_result$model_summary)
print(il_bf_result$plot)

bla_bf_result <- process_data_and_plot(data,  "bla_bf", "theta_1")
print(bla_bf_result$model_summary)
print(bla_bf_result$plot)

bla_il_result <- process_data_and_plot(data,  "bla_il", "theta_2")
print(bla_il_result$model_summary)
print(bla_il_result$plot)


il_bf_result <- process_data_and_plot(il_bf_data, 'il_bf', "theta_2")
print(il_bf_result$model_summary)
print(il_bf_result$plot)

bla_bf_result <- process_data_and_plot(data,  "bla_bf", "theta_2")
print(bla_bf_result$model_summary)
print(bla_bf_result$plot)



