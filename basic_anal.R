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
in  } else {
    data$stereotrode <- factor(data$stereotrode)
    }
  
  data$condition <- factor(data$condition)
  
  # Reorder time_period factor levels
  data$period <- factor(data$period, 
                             levels = c("pre_tone", 
                                        "post_beep_0_300", 
                                        "post_beep_301_600", 
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


pn_unit_model = lmer(spike_rate ~ condition*period + (1|animal/unit_num), data = pn_data)
in_unit_model = lmer(spike_rate ~ condition*period + (1|animal/unit_num), data = in_data)
pn_results = tab_model(pn_unit_model)
in_results = tab_model(in_unit_model)

residuals = resid(in_unit_model)
highest_values <- sort(residuals, decreasing = TRUE)[1:3]

in_data_no_residuals = in_data[-c(118, 298)]
in_no_resid_model =  lmer(spike_rate ~ condition*period + (1|animal/unit_num), data = in_data_no_residuals)
in_no_resid_boxplot <- ggplot(data = in_no_resid_data, mapping = aes(x = condition, y = spike_rate, fill = period)) + 
  geom_boxplot() +
  facet_wrap(~ unit_type, ncol = 1)
in_no_resid_results = tab_model(in_no_resid_model)

png(filename=file.path(graph_dir, "stereotrode_boxplot.png"), width=800, height=600)
stereotrode_graph <- ggplot(data = stereotrode_data, mapping = aes(x = condition, y = spike_rate, fill = period)) + 
  geom_boxplot() +
  facet_wrap(~ unit_type, ncol = 1)
dev.off()

print(stereotrode_graph)


png(filename=file.path(graph_dir, "pn_boxplot.png"), width=800, height=600)
# Create the box plot without original outlier points
pn_box_plot <- ggplot(data = pn_data, mapping = aes(x = condition, y = spike_rate, fill = period)) + 
  geom_boxplot() +
  facet_wrap(~ unit_type, ncol = 1)

dev.off()

png(filename=file.path(graph_dir, "in_boxplot.png"), width=800, height=600)
in_box_plot <- ggplot(data = in_data, mapping = aes(x = condition, y = spike_rate, fill = period)) + 
  geom_boxplot() +
  facet_wrap(~ unit_type, ncol = 1)
dev.off()





