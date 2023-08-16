library(glmmTMB)
library(lme4)
library(lmerTest)  # for lmer p-values
library(readr)
library(sjPlot)
library(ggplot2)
library(broom)
library(emmeans)
library(dplyr)

csv_file = '/Users/katie/likhtik/data/lfp/lfp_theta_1_lfp_theta_2_lfp_delta_pl_period_frequency_periods.csv'

# Slide 20

prepare_df <- function(csv){
  df <- read_csv(csv)
  
  # Convert variables to factors
  factor_vars <- c('animal', 'group', 'period_type')
  df[factor_vars] <- lapply(df[factor_vars], factor)
  
  return(df)
}

means_df <- subset_df %>%
  group_by(group, period, period_type) %>%
  summarize(mean_value = mean(theta_1_power, na.rm = TRUE))

df = prepare_df(csv_file)

subset_df <- subset(df, period %in% c(0, 4))

slide_20_model <- lmer(theta_1_power ~ group*period*period_type + (1|animal), data=subset_df)
summary(slide_20_model)

plot <- emmip(slide_20_model, group ~ period_type | period, 
              at = list(period = c(0, 4)),CIs = FALSE)

ggplot(means_df, aes(x = period_type, y = mean_value, fill = period_type)) +
  geom_bar(stat="identity", position="dodge") +
  facet_grid(group ~ period, scales = "free") +
  labs(title = "Bar graph of raw means", x = "Period Type", y = "Mean Value") +
  theme_minimal()

### Continuous plot of Frequencies 0-14 in periods 0 and 4
df <- read.csv("/Users/katie/likhtik/data/lfp/lfp_delta_theta_pl_period_frequency_periods.csv")

# Aggregate data by frequency_bin, period_type, group, and period to get mean delta_theta_power
df_agg <- df %>% 
  group_by(frequency_bin, period_type, group, period) %>% 
  summarise(mean_delta_theta_power = mean(delta_theta_power))

# Filter the aggregated data to only include period 0 or 4, and group "control" or "stressed"
df_subset <- df_agg %>% filter(period %in% c(0, 4) & group %in% c("control", "stressed"))

# Create the plot
plot <- ggplot(data = df_subset, aes(x = frequency_bin, y = mean_delta_theta_power, color = period_type)) + 
  geom_line() + 
  geom_point() +
  facet_grid(group ~ period, scales = "free", labeller = labeller(period = c(`0` = "Period: 0", `4` = "Period: 4"))) + 
  scale_color_manual(values = c("pretone" = "gray", "tone" = "blue")) + 
  labs(x = "Frequency (Hz)", y = "Power AU", color = "Period Type") + 
  theme_minimal() + 
  theme(legend.position = "bottom")

# Display the plot
print(plot)

### Slide 21



csv_file = '/Users/katie/likhtik/data/lfp/lfp_theta_1_lfp_theta_2_lfp_delta_pl_period_frequency_periods.csv'

df = read.csv(csv_file)

# Calculate the ratio of theta_1_power to delta_power
df$theta_delta_ratio <- df$theta_1_power / df$delta_power

# Aggregate data by period, period_type, and group to get mean theta/delta ratio
df_agg <- df %>% 
  group_by(period, period_type, group) %>% 
  summarise(
    mean_theta_delta_ratio = mean(theta_delta_ratio),
    se_theta_delta_ratio = sd(theta_delta_ratio) / sqrt(n())
  )

# Create the plot
plot <- ggplot(data = df_agg, aes(x = as.factor(period), y = mean_theta_delta_ratio)) + 
  geom_point(aes(shape = period_type, color = period_type), size = 3) + 
  geom_errorbar(aes(ymin = mean_theta_delta_ratio - se_theta_delta_ratio, 
                    ymax = mean_theta_delta_ratio + se_theta_delta_ratio, 
                    color = period_type), width = 0.2) +
  scale_shape_manual(values = c("pretone" = 15, "tone" = 6)) + 
  scale_color_manual(values = c("pretone" = "gray", "tone" = "blue")) + 
  facet_grid(rows = vars(group), labeller = labeller(group = c(`control` = "Control", `stressed` = "Stressed"))) + 
  labs(x = "Period", y = "Theta/Delta Ratio", color = "Period Type", shape = "Period Type") + 
  theme_minimal() + 
  theme(legend.position = "bottom")

# Display the plot
print(plot)

### Delta power

# Aggregate data by period, period_type, and group to get mean delta_power and standard error
df_agg <- df %>% 
  group_by(period, period_type, group) %>% 
  summarise(
    mean_delta_power = mean(delta_power),
    se_delta_power = sd(delta_power) / sqrt(n())
  )

# Create the plot
plot <- ggplot(data = df_agg, aes(x = as.factor(period), y = mean_delta_power, group = period_type)) + 
  geom_line(aes(color = period_type)) +
  geom_point(aes(shape = period_type, color = period_type), size = 3) + 
  geom_errorbar(aes(ymin = mean_delta_power - se_delta_power, 
                    ymax = mean_delta_power + se_delta_power, 
                    color = period_type), width = 0.2) +
  scale_shape_manual(values = c("pretone" = 15, "tone" = 6)) + 
  scale_color_manual(values = c("pretone" = "gray", "tone" = "blue")) + 
  facet_grid(rows = vars(group), labeller = labeller(group = c(`control` = "Control", `stressed` = "Stressed"))) + 
  labs(x = "Period", y = "Delta Power", color = "Period Type", shape = "Period Type") + 
  theme_minimal() + 
  theme(legend.position = "bottom")

# Display the plot
print(plot)
