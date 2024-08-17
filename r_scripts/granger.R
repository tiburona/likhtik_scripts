library(lme4)
library(dplyr)
library(glmmTMB)
library(lmerTest)
library(sjPlot)
library(ggplot2)
library(emmeans)
library(DHARMa)
library(gamlss)


csv_name = 'granger.csv'
granger_csv = paste('/Users/katie/likhtik/IG_INED_Safety_Recall/granger_f_stat', csv_name, sep='/')
granger_df <- read.csv(granger_csv, comment.char="#") 


# Convert animal, group, and period_type to factors
factor_vars <- c('animal', 'group', 'period_type', 'granger_calculator')
granger_df[factor_vars] <- lapply(granger_df[factor_vars], factor) 
# Example data frame


# Find and rename the column
granger_df <- granger_df %>%
  rename_with(
    .fn = ~ "forward",
    .cols = contains("forward")
  ) %>%
  rename_with(
    .fn = ~ "backward",
    .cols = contains("backward")
  )

df1 <- granger_df[!is.na(granger_df$forward), ]

# Step 2: Separate the rows where Column2 has data
df2 <- granger_df[!is.na(granger_df$backward), ]

# Step 3: Merge the two data frames based on the key columns
merged_df <- merge(df1, df2, 
                   by = c('frequency', 'frequency_bin', 'period_type', 'animal', 
                          'group', 'granger_calculator'), all = TRUE)

# This will create columns named 'Column1.x' and 'Column2.y'. 
# Step 4: Rename columns if necessary
merged_df <- merged_df %>%
  rename(forward = forward.x, backward = backward.y)







# a different command you can use to filter/subset data
granger_data <- subset(merged_df, merged_df$frequency_bin <= 8)
granger_data <- subset(merged_df, merged_df$frequency_bin >= 0)
granger_data_bla_pl <- granger_data %>% 
  group_by(animal, period_type, granger_calculator, group) %>%
  summarize(bla_pl_theta_1_granger = log(mean(forward, na.rm=TRUE)),
            pl_bla_theta_1_granger = log(mean(backward, na.rm=TRUE)), .groups='drop')
  
granger_data_bla_pl$diff <- granger_data_bla_pl$bla_pl_theta_1_granger - granger_data_bla_pl$pl_bla_theta_1_granger

bla_pl_theta_1_model <-  lmer(bla_pl_theta_1_granger ~ group * period_type + (1|animal), 
                          data <- granger_data_bla_pl)

summary(bla_pl_theta_1_model)

hist(granger_data_bla_pl$bla_pl_theta_1_granger)

residuals <- resid(bla_pl_theta_1_model)

hist(residuals, main = "Histogram of Residuals")

plot(residuals ~ fitted(bla_pl_theta_1_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

qqnorm(residuals, main = "Q-Q Plot of Residuals")
qqline(residuals, col = "red")

