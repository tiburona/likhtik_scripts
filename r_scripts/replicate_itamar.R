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


df <- read.csv("/Users/katie/likhtik/data/lfp/lfp_delta_theta_pl_period_frequency_periods.csv")

# Filter the data to only include period 0 or 4, and group "control" or "stressed"
df_subset <- df[df$period %in% c(0, 4) & df$group %in% c("control", "stressed"), ]

# Create the plot
plot <- ggplot(data = df_subset, aes(x = frequency_bin, y = delta_theta_power, color = period_type)) + 
  geom_line(aes(linetype = period_type)) + 
  facet_grid(group ~ period, scales = "free", labeller = labeller(period = c(`0` = "Period: 0", `4` = "Period: 4"))) + 
  scale_color_manual(values = c("pretone" = "gray", "tone" = "blue")) + 
  labs(x = "Frequency (Hz)", y = "Power AU", color = "Period Type") + 
  theme_minimal() + 
  theme(legend.position = "bottom")
