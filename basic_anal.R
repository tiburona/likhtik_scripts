library(lme4)

library(ggplot2)
library(ggeffects)
library(sjPlot)
library(plotly)
library(dplyr)

data_dir <- '/Users/katie/likhtik/data'
graph_dir = file.path('data_dir', 'graphs')


process_data <- function(element_type, data_dir) {
  
  data_file <- file.path(data_dir, paste('firing_rates_by_', element_type, '.csv', sep=""))
  # Read the CSV filed
  data <- read.csv(data_file, row.names = NULL)
  
  # Convert columns to factors
 
  data$animal <- factor(data$animal)
  
  
  if (element_type == 'unit') {
    data$unit_num <- factor(data$unit_num)
  } else {
    data$stereotrode <- factor(data$stereotrode)
    }
  
  data$condition <- factor(data$condition)
  data$category <- factor(data$category)
  
  # Reorder time_period factor levels
  data$period <- factor(data$period, 
                             levels = c("late_post_beep",
                                        "during_beep", 
                                        "early_post_beep", 
                                        "mid_post_beep"))
  
  # Remove rows with empty condition values
  data <- subset(data, condition != "")
  
  return(data)
}

unit_data <- process_data('unit', data_dir)
unit_model = lmer(rate ~ condition*category*period + (1|animal/unit_num), data=unit_data)
png(filename=file.path(graph_dir, "pn_in_boxplot.png"), width=800, height=600)
unit_graph <- ggplot(data = unit_data, 
                     mapping = aes(x = factor(category, levels = c("PN", "IN")), 
                                   y = rate, 
                                   fill = factor(period, levels = c("during_beep", 
                                                                    "early_post_beep", 
                                                                    "mid_post_beep", 
                                                                    "late_post_beep")))) + 
  geom_boxplot() +
  facet_wrap(~ condition, ncol = 1)


dev.off()

tab_model(unit_model)
