library(lme4)

library(ggplot2)
library(ggeffects)
library(sjPlot)

data_file = '/Users/katie/likhtik/data/firing_rates.csv'
data = read.csv(data_file)

data$unit_num = factor(data$unit_num)
data$animal = factor(data$animal)
data$unit_type = factor(data$unit_type)
data$condition = factor(data$condition)
data$time_period <- factor(data$time_period, levels = c("pre-tone", "post-beep (0-600 ms)", "late post-beep (>600 ms)", "inter-tone"))
data <- subset(data, condition != "")

good_data = subset(data, data$unit_type == 'good')
fit = lmer(spike_rate ~ condition*unit_type*time_period + (1|animal/unit_num), data=data)

model <- aov(count ~ condition*unit_type + Error(animal), data = data)

result <- t.test(count ~ condition, data = data)

# create effects plot for interaction of unit_type and condition
interaction_plot <- ggpredict(fit, terms = c("unit_type", "condition"), type = "fe")


p <- ggplot(data = data, mapping = aes(x = condition, y = spike_rate, fill = time_period)) + 
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
