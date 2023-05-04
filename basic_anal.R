library(lme4)

library(ggplot2)
library(ggeffects)
library(sjPlot)

data_dir <- '/Users/katie/likhtik/data'


process_data <- function(element_type, data_dir) {
  
  data_file <- file.path(data_dir, paste('firing_rates_by_', element_type, '.csv', sep=""))
  # Read the CSV filed
  data <- read.csv(data_file)
  
  # Convert columns to factors
 
  data$unit_type <- factor(data$unit_type)
  data$animal <- factor(data$animal)
  
  if (element_type == 'unit') {
    data$unit_num <- factor(data$unit_num)
  } else {
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
stereotrode_data <- process_data('stereotrode', data_dir)


stereotrode_model = lmer(spike_rate ~ condition*unit_type*period + (1|animal/stereotrode), data=stereotrode_data)

good_unit_data = subset(unit_data, unit_data$unit_type == 'good')

unit_model = lmer(spike_rate ~ condition*period + (1|animal/unit_num), data = good_unit_data)

# model <- aov(spike_rate ~ condition*unit_type + Error(animal), data = data)


# create effects plot for interaction of unit_type and condition
interaction_plot <- ggpredict(fit, terms = c("unit_type", "condition"), type = "fe")


p <- ggplot(data = stereotrode_data, mapping = aes(x = condition, y = spike_rate, fill = period)) + 
  geom_boxplot() +
  facet_wrap(~ unit_type, ncol = 1)

print(p)

# create bar graph with standard error bars
# ggplot(interaction_plot, aes(x = interaction_term, y = predicted, ymin = conf.low, ymax = conf.high)) +
#   geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
#   geom_errorbar(position = position_dodge(width = 0.9), width = 0.2) +
#   xlab("Unit Type x Condition") + ylab("Count") +
#   ggtitle("Count by Unit Type and Condition") +
#   theme_bw()
