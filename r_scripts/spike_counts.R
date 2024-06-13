library(lme4)
library(lmerTest)
library(emmeans)
library(ggplot2)
library(dplyr)
library(DHARMa)
library(glmmTMB)


# Load the CSV file
count_data <- 
  read.csv('/Users/katie/likhtik/IG_INED_Safety_Recall/spike_counts/count_spreadsheet.csv', 
           comment.char="#")


count_data <- subset(count_data, count_data$time_bin < 30)

count_data <- count_data %>%
  group_by(group, neuron_type, period_type, animal, unit, period, event) %>%
  summarise(count = sum(spike_counts, na.rm = TRUE), .groups = 'drop')


# Make sure every categorical variable is a factor
factor_vars <- c('animal', 'group', 'period_type', 'neuron_type', 'unit')
count_data[factor_vars] <- lapply(count_data[factor_vars], factor)



count_data_model_poisson <- glmer(count ~ group * period_type * neuron_type + 
                                  (1|animal:unit) + (1|animal:unit:period), 
                                family = poisson(link = "log"), 
                                data = count_data)

summary(count_data_model_poisson)

emm_results = emmeans(count_data_model_poisson, specs = pairwise ~ group | period_type | neuron_type)
emm_means <- summary(emm_results, type = "response")  # This directly gives you the response scale values

count_data_plot <- emmip(emm_results, group ~ period_type | neuron_type, CIs = FALSE, type = "response") +
  labs(y = "Predicted Count of Spikes per Event (0-.3s)") +
  scale_color_manual(values = c("control" = "purple", "defeat" = "orange"))

print(count_data_plot)

simulationOutput <- simulateResiduals(fittedModel = count_data_model_poisson)
plot(simulationOutput)
testDispersion(simulationOutput)

zeroInflationTest <- testZeroInflation(simulationOutput)
print(zeroInflationTest)

predictions <- predict(count_data_model_poisson, type = "response")
summary(predictions)


# I experimented with a zero-inflated model but decided Poisson was good enough.

zip_model <- glmmTMB(count ~ group * period_type * neuron_type + (1|animal:unit) + (1|animal:unit:period),
                     ziformula = ~ 1,  # Zero-inflation part
                     family = poisson(link = "log"),
                     data = count_data)
summary(zip_model)

AIC(zip_model)
AIC(count_data_model_poisson)  # 

zip_emm_results = emmeans(zip_model, specs = pairwise ~ group | period_type | neuron_type)
zip_emm_means <- summary(zip_emm_results, type = "response")  # This directly gives you the response scale values

zip_data_plot <- emmip(zip_emm_results, group ~ period_type | neuron_type, CIs = FALSE, type = "response") +
  labs(y = "Predicted Count of Spikes per Event (0-.3s)") +
  scale_color_manual(values = c("control" = "purple", "defeat" = "orange"))

print(zip_data_plot)

simulated_res_zip = simulateResiduals(zip_model)
plot(simulated_res_zip)
testOutliers(simulated_res_zip)
testDispersion(simulated_res_zip)


## post hocs

count_data_IN_tone = subset(count_data, count_data$neuron_type=='IN' & count_data$period_type=='tone')

IN_tone_model_poisson <- glmer(count ~ group + 
                                    (1|animal/unit/period), 
                                  family = poisson(link = "log"), 
                                  data = count_data_IN_tone)

summary(IN_tone_model_poisson)


count_data_IN_pretone = subset(count_data, count_data$neuron_type=='IN' & count_data$period_type=='pretone')

IN_pretone_model_poisson <- glmer(count ~ group + 
                                 (1|animal/unit/period), 
                               family = poisson(link = "log"), 
                               data = count_data_IN_pretone)

summary(IN_pretone_model_poisson)


count_data_PN_tone = subset(count_data, count_data$neuron_type=='PN' & count_data$period_type=='tone')

PN_tone_model_poisson <- glmer(count ~ group + 
                                 (1|animal:unit) + (1|animal:unit:period), 
                               family = poisson(link = "log"), 
                               data = count_data_PN_tone)

summary(PN_tone_model_poisson)


count_data_PN_pretone = subset(count_data, count_data$neuron_type=='PN' & count_data$period_type=='pretone')

PN_pretone_model_poisson <- glmer(count ~ group + 
                                    (1|animal:unit) + (1|animal:unit:period), 
                                  family = poisson(link = "log"), 
                                  data = count_data_PN_pretone)

summary(PN_pretone_model_poisson)










