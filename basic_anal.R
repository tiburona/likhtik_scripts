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
 
  data$unit_type <- factor(data$unit_type)
  data$animal <- factor(data$animal)

  
  if (element_type == 'unit') {
    data$unit_num <- factor(data$unit_num)
    data$category <- factor(data$category)
    } else {
    data$stereotrode <- factor(data$stereotrode)
    }
  
  data$condition <- factor(data$condition)
  
  # Reorder time_period factor levels
  data$period <- factor(data$period, 
                             levels = c("pre_tone", 
                                        "during_beep",
                                        "early_post_beep",
                                        "mid_post_beep",
                                        "late_post_beep", 
                                        "inter_tone"))
  
  # Remove rows with empty condition values
  data <- subset(data, condition != "")
  
  return(data)
}

unit_data <- process_data('unit', data_dir)
pn_data <- subset(unit_data, unit_data$category == 'PN')
in_data <- subset(unit_data, unit_data$category == 'IN')
stereotrode_data <- process_data('stereotrode', data_dir)

stereotrode_model = lmer(spike_rate ~ condition*unit_type*period + (1|animal/stereotrode), data=stereotrode_data)
tab_model(stereotrode_model)

unit_model = lmer(spike_rate ~ condition*category*period + (1|animal), data=unit_data)

png(filename=file.path(graph_dir, "pn_in_boxplot.png"), width=800, height=600)
unit_graph <- ggplot(data = unit_data, mapping = aes(x = condition, y = spike_rate, fill = period)) + 
  geom_boxplot() +
  facet_wrap(~ category, ncol = 1)
dev.off()

print(unit_graph)
tab_model(unit_model)








