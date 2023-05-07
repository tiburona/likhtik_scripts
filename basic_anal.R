library(lme4)

library(ggplot2)
library(ggeffects)
library(sjPlot)
library(plotly)
library(dplyr)

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
tab_model(stereotrode_model)

good_unit_data = subset(unit_data, unit_data$unit_type == 'good')

unit_model = lmer(spike_rate ~ condition*period + (1|animal/unit_num), data = good_unit_data)
tab_model(unit_model)

# model <- aov(spike_rate ~ condition*unit_type + Error(animal), data = data)


# create effects plot for interaction of unit_type and condition
# interaction_plot <- ggpredict(fit, terms = c("unit_type", "condition"), type = "fe")


p <- ggplot(data = stereotrode_data, mapping = aes(x = condition, y = spike_rate, fill = period)) + 
  geom_boxplot() +
  facet_wrap(~ unit_type, ncol = 1)

print(p)



q <- ggplot(data = good_unit_data, mapping = aes(x = condition, y = spike_rate, fill = period)) + 
  geom_boxplot()

# Create the box plot without original outlier points
q <- ggplot(data = good_unit_data, mapping = aes(x = condition, y = spike_rate, fill = period)) + 
  geom_boxplot(outlier.shape = NA)

# Add points with hover text
q <- q + geom_point(aes(text = paste("Animal:", animal, "<br>Unit_num:", unit_num)),
                    position = position_jitter(width = 0.2, height = 0),
                    alpha = 0.5)

# Convert the ggplot object to a plotly object
interactive_q <- ggplotly(q, tooltip = c("text"))

# Display the interactive plot
interactive_q

# Load the required libraries
# Create the box plot without original outlier points
q <- ggplot(data = good_unit_data, mapping = aes(x = condition, y = spike_rate, fill = period)) + 
  geom_boxplot(outlier.shape = NA) +
  facet_wrap(~condition, ncol = 5, scales = "free_x")

# Calculate the box plot statistics
boxplot_stats <- good_unit_data %>% 
  group_by(condition, period) %>% 
  summarise(lower = quantile(spike_rate, 0.25) - 1.5 * IQR(spike_rate),
            upper = quantile(spike_rate, 0.75) + 1.5 * IQR(spike_rate))

# Filter the outliers
outliers <- good_unit_data %>%
  left_join(boxplot_stats, by = c("condition", "period")) %>%
  filter(spike_rate < lower | spike_rate > upper)

# Add interactive points with hover text
q <- q + geom_point(data = outliers,
                    aes(text = paste("Animal:", animal, "<br>unit num:", unit_num)),
                    position = position_jitter(width = 0.05, height = .05),
                    alpha = 0.5)

# Convert the ggplot2 object to a plotly object
interactive_plot <- ggplotly(q, tooltip = "text")

# Display the interactive plot
interactive_plot
# create bar graph with standard error bars
# ggplot(interaction_plot, aes(x = interaction_term, y = predicted, ymin = conf.low, ymax = conf.high)) +
#   geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
#   geom_errorbar(position = position_dodge(width = 0.9), width = 0.2) +
#   xlab("Unit Type x Condition") + ylab("Count") +
#   ggtitle("Count by Unit Type and Condition") +
#   theme_bw()
