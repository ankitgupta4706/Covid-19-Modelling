# Covid-19-Modelling
The SEIRV model is a popular epidemiological model used for COVID-19 forecasting and analysis. This model divides the population into different compartments, namely Susceptible, Exposed, Infected, Recovered, and Vaccinated, to better understand the spread of the virus and the impact of different interventions. The SEIRV model is widely used by researchers and policymakers around the world to make informed decisions related to COVID-19.

# Methodology
We use the SEIRV model with data of first dose vaccinations and immunity waning to model and predict
COVID-19 cases for Karnatake state. We fine tune the parameters (Beta, S0, E0, I0, R0 and CIR0) to fit the model to the number of
COVID cases between 16th March 2021 to 26th April 2021, and we use those fitted parameters to make future
predictions and do various analysis with respect to the contact rate Beta. CIR is cases to infections ratio.

# Data
1. The cumulative number of reported cases from 16th March 2021 until 20 September 2021.
2. The cumulative number of tests done (column name: tested) from 16th March 2021 until 20 September 2021.
3. The cumulative number of vaccinations (dose 1) administered (column name: First Dose Administered) since 16 January 2021 until 20 September 2021.

# Model Equations
<img width="372" alt="Covid_Model" src="https://user-images.githubusercontent.com/81372735/236662718-9e94f2b9-25cd-444d-a8e4-f0cc7440d59a.PNG">

# Assumptions
- Total Population (N): 70 million
- Mean Incubation Period (1/alpha): 5.8 days
- Mean Recovery Period (1/gamma): 5 days
- R(0) is between 15.6% and 36% of the population.
- The initial cases-to-infections ratio (CIR(0)) is between 12.0 and 30.0.
- Vaccine Efficacy: 66%
- Delta W(t) = R(0) / 30 for t between 16 March 12021 and 15 April 2021
- Delta W(t) = Delta R(t-180) + Epsilon * Delta V(t - 180) when t is larger than 11 September 2021
- define CIR(t) = CIR(0) * T(t0) / T(t)
- 2 lakhs vaccinations per day starting 27 April 2021

# Loss Funtion 
<img width="280" alt="loss_function" src="https://user-images.githubusercontent.com/81372735/226107218-65188be7-4900-4a0d-a42c-d3d1fbd5a377.PNG">
where \bar{c(t)} is the seven day average of daily confiermed cases and similarly, \bar{e(t)} is the running seven day average
of estimated cases.

# Results

*Optimum Values*

Optimum_loss is 0.00910590987196208
Optimum parameters are mentioned below:

- beta is  0.48999738492093337
- S is  48789999.999997385
- E is  139999.99999738493
- I is  69999.99999738495
- R is  20999999.999997385
- CIR is  18.499997384920935

*Open Loop Control*


For various contact rate (beta)

*Closed Loop Control*


<img width="284" alt="closed_loop_control" src="https://user-images.githubusercontent.com/81372735/226107697-9a8a65a0-fc47-47e1-a6db-c98f49d56f07.PNG">

*Plots*
Infections per day VS Time
![Figure_1](https://user-images.githubusercontent.com/81372735/226107735-1de31c81-f56d-406f-a589-0492db552c73.png)

Fraction of Susceptible Population Vs Time
![Figure_2](https://user-images.githubusercontent.com/81372735/226107738-65cb2879-0fd4-41c0-b464-5ad7d9318209.png)

# Observations

As expected, the new daily cases are highest for BETA, followed by 2/3 BETA, 1/2 BETA and 1/3
BETA. The graph for Closed-Loop control is also shown above in Purple color. Higher BETA values
means that due to high contact rate, more people will be infected, but due to the rapid growth of
the exposed, infected and recovered population, the curve also bends down very quickly because
susceptible population also reduced quickly for larger BETA values.
