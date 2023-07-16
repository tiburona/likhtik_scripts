library(lme4)
library(readr)

df <- read_csv('/Users/katie/likhtik/data/proportion_score_continuous_units.csv')

# Convert variables to factors
factor_vars <- c('unit_num', 'animal', 'category', 'condition')
df[factor_vars] <- lapply(df[factor_vars], factor)

# Run logistic regression
model <- glmer(proportion ~ category * condition + (1 | animal/unit_num), family = binomial, data = df)
summary(model)

