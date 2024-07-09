library(dplyr)
library(tidyr)
library(ggplot2)
library(lme4)
library(stringr)

csv = paste('/Users/katie/likhtik/CH_EXT/phase_phase_mrl', 'phase_phase.csv', sep='/')
data <- read.csv(csv, comment.char="#") 
read_metadata(csv)

data$bla_bf_five_phase_phase_mrl <- data$bla_bf__5_.6__phase_phase_mrl 
data$bf_il_five_phase_phase_mrl <- data$bf_il__5_.6__phase_phase_mrl 


bla_il_csv = paste('/Users/katie/likhtik/CH_EXT/phase_phase_mrl', 'phase_phase_bla_il.csv', sep='/')
bla_il_data <- read.csv(bla_il_csv, comment.char="#") 
data$bla_il_five_phase_phase_mrl <- data$bla_il__5_.6__phase_phase_mrl 


process_data_and_plot <- function(data, region_set, frequency_band) {
  regions <- unlist(strsplit(region_set, "_"))
  region1 <- toupper(regions[1])
  region2 <- toupper(regions[2])
  frequency_band_formatted <- paste(tools::toTitleCase(gsub("_", " to ", frequency_band)), "Hz")  # Capitalize and replace underscores
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
  
  grouped_data <- clean_data %>%
    group_by(period_type, animal, period, time) %>%
    summarise(
      mrl = mean(.data[[variable_name]], na.rm = TRUE),
      .groups = "drop"
    )
  
  result_data <- grouped_data %>%
    filter(period_type %in% c("pretone", "tone")) %>%
    pivot_wider(
      id_cols = c(animal, period, time),
      names_from = period_type,
      values_from = mrl
    ) %>%
    group_by(animal, time) %>%
    mutate(difference = tone - pretone) %>%
    distinct(animal, .keep_all = TRUE) %>%
    select(animal, difference)
  
  # Statistical modeling
  model_formula <- as.formula(paste(variable_name, " ~ time * period_type + (1|animal)"))
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
    scale_fill_manual(values = c("early" = "#82086F", "late" = "#D52C90")) +
    scale_x_discrete(labels = c("early" = "Periods 1 & 2", "late" = "Periods 19 & 20"))
  
  # Return list containing the model summary and the plot
  list(model_summary = model_summary, plot = plot)
}


process_data_and_plot_tone <- function(data, region_set, frequency_band) {
  regions <- unlist(strsplit(region_set, "_"))
  region1 <- toupper(regions[1])
  region2 <- toupper(regions[2])
  frequency_band_formatted <- paste(tools::toTitleCase(gsub("_", " to ", frequency_band)), "Hz")  # Capitalize and replace underscores
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
    filter(period %in% c(0, 1, 18, 19)) %>%
    filter(period_type == 'pretone')
  
  grouped_data <- clean_data %>%
    group_by(period_type, animal, period, time) %>%
    summarise(
      mrl = mean(.data[[variable_name]], na.rm = TRUE),
      .groups = "drop"
    )
  

  
  # Statistical modeling
  model_formula <- as.formula(paste(variable_name, " ~ time + (1|animal)"))
  model <- lmer(model_formula, data = clean_data)
  model_summary <- summary(model)
  
  # Calculate means and standard errors for plotting
  graph_data <- grouped_data %>%
    group_by(time) %>%
    summarise(
      mean = mean(mrl, na.rm = TRUE),
      sd = sd(mrl, na.rm = TRUE),
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
      title = sprintf("Tone Phase-Phase %s MRL Between %s and %s", frequency_band_formatted, region1, region2),
      x = NULL,  # No x-axis label
      y = "MRL"
    ) +
    theme_minimal() +
    scale_fill_manual(values = c("early" = "#82086F", "late" = "#D52C90")) +
    scale_x_discrete(labels = c("early" = "Periods 1 & 2", "late" = "Periods 19 & 20"))
  
  # Return list containing the model summary and the plot
  list(model_summary = model_summary, plot = plot)
}

# 
# bla_il_result <- process_data_and_plot(data,  "bla_il", "theta_1")
# print(bla_il_result$model_summary)
# print(bla_il_result$plot)
# 
# 
# il_bf_result <- process_data_and_plot(il_bf_data, 'il_bf', "theta_1")
# print(il_bf_result$model_summary)
# print(il_bf_result$plot)





bla_bf_result <- process_data_and_plot(data,  "bla_bf", "five")
print(bla_bf_result$model_summary)
print(bla_bf_result$plot)

bf_il_result_tone <- process_data_and_plot(data,  "bf_il", "five")
print(bf_il_result_tone$model_summary)
print(bf_il_result_tone$plot)

bla_il_result <- process_data_and_plot(bla_il_data,  "bla_il", "five")
print(bla_il_result$model_summary)
print(bla_il_result$plot)


il_bf_result <- process_data_and_plot(il_bf_data, 'il_bf', "theta_2")
print(il_bf_result$model_summary)
print(il_bf_result$plot)

bla_bf_result <- process_data_and_plot(data,  "bla_bf", "theta_2")
print(bla_bf_result$model_summary)
print(bla_bf_result$plot)


process_data_and_plot_non_evoked <- function(data, region_set, frequency_band) {
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
      time = as.factor(time),
      period_label = case_when(
        time == "early" & period_type == "pretone" ~ "Pretone Periods 1 & 2",
        time == "early" & period_type == "tone" ~ "Tone Periods 1 & 2",
        time == "late" & period_type == "pretone" ~ "Pretone Periods 19 & 20",
        time == "late" & period_type == "tone" ~ "Tone Periods 19 & 20"
      )
    ) %>%
    filter(period %in% c(0, 1, 18, 19), period_type %in% c("pretone", "tone"))
  
  # Calculate means and standard errors
  summary_data <- clean_data %>%
    group_by(period_label) %>%
    summarise(
      mean = mean(.data[[variable_name]], na.rm = TRUE),
      se = sd(.data[[variable_name]], na.rm = TRUE) / sqrt(n()),
      n = n()
    )
  
  # Plotting
  plot <- ggplot(summary_data, aes(x = period_label, y = mean, fill = period_label)) +
    geom_bar(stat = "identity", position = position_dodge(), width = 0.7) +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.2, position = position_dodge(0.7)) +
    scale_fill_manual(values = c(
      "Pretone Periods 1 & 2" = "darkgray", 
      "Tone Periods 1 & 2" = "#82086F", 
      "Pretone Periods 19 & 20" = "lightgray", 
      "Tone Periods 19 & 20" = "#D52C90"
    )) +
    labs(
      title = sprintf("Phase-Phase %s MRL Between %s and %s", frequency_band_formatted, region1, region2),
      y = "MRL"
    ) +
    theme_minimal() +
    theme(axis.title.x = element_blank(), legend.position = "none")
  
  # Return the plot
  plot
}




bla_il_plot <- process_data_and_plot_non_evoked(data,  "bla_il", "theta_2")
bla_bf_plot <- process_data_and_plot_non_evoked(data,  "bla_bf", "theta_2")
il_bf_plot <- process_data_and_plot_non_evoked(il_bf_data,  "il_bf", "theta_2")
