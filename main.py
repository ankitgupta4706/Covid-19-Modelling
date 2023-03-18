import numpy as np
import pandas as pd
import scipy.optimize
import datetime
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None


"gets the actual no. of reported cases per day "
def get_reported_cases(df):
    del_confirmed = []
    dates = []
    date = df['Date'].iloc[0]

    for i in range(len(df)):
        dates.append(date)
        if i==0:
            del_confirmed.append(int(df['Confirmed'].iloc[i]) - base_confirmed_case)
        else:
            del_confirmed.append(int(df['Confirmed'].iloc[i]) - int(df['Confirmed'].iloc[i-1]))
        date += datetime.timedelta(days=1)
    return  del_confirmed, dates
    

"Important Function, takes the following parameters"
"Description of Input:"
"1: params = [beta0, S0, E0, I0, R0, CIR0]"
"2: df = dataframe"
"3: prediction= if prediction is set as False returns the error for a given initialization, else"
"...if set as True, does prediction till 31st december, 2021 and returns the data"
"4: frac_beta = the fraction with which beta should be multiplied if open-loop control is used, works only when open loop is given"
"2: control = open loop or closed loop, works only when prediction is set as True"

def calculate_loss(params, df, prediction = False, frac_beta = 1, control = 'open loop'):
    beta = params[0]
    S= params[1]
    E = params[2]
    I = params[3]
    R = params[4]
    CIR = params[5]
    S_array = [S]
    
    CIR0 = CIR
    R0 = R
    I0 = I
    eps = 0.66
    alpha = 1/5.8
    gamma = 1/5
    N = 70000000
    infection_params = np.zeros(len(df))
    del_confirmed = np.zeros(len(df))
    del_infection_params = np.zeros(len(df))
    
    bar_del_confirmed = np.zeros(len(df))
    bar_del_infection_params = np.zeros(len(df)) 
    
    date_array_prediction = []
    infection_array_prediction = []
    
    delR_array = []
    delV_array = []
    S_array = []
    
    T_t0 = df['Tested'].iloc[0]
    for i in range(len(df)):
        
        if (df['Date'].iloc[i] >= pd.to_datetime('16-03-2021', infer_datetime_format=True)) and (df['Date'].iloc[i] <= pd.to_datetime('15-04-2021', infer_datetime_format=True)):
            delW = R0/30
        else:
            delW = 0
        
        if (df['Date'].iloc[i] >= pd.to_datetime('16-03-2021', infer_datetime_format=True)) and (df['Date'].iloc[i] <= pd.to_datetime('22-03-2021', infer_datetime_format=True)):
            T_t = sum(df['Tested'].iloc[:i+1])/len(df['Tested'].iloc[:i+1])
        else:
            T_t = sum(df['Tested'].iloc[i-6:i+1])/len(df['Tested'].iloc[i-6:i+1])
            
        if i==0:
            delV = int(df['First Dose Administered'].iloc[i]) - base_vaccinated_case
        else:
            delV = int(df['First Dose Administered'].iloc[i]) - int(df['First Dose Administered'].iloc[i-1])
            
        delS = - (beta * S * I / N) -  eps * delV + delW
        delE =   (beta * S * I / N) -  alpha * E
        delI =   alpha * E - gamma * I
        delR =   gamma * I + eps * delV - delW 
        CIR  =   CIR0*T_t0/T_t
        del_infection_params[i] = alpha * E / CIR
        
        S = S + delS
        E = E + delE
        I = I + delI
        R = R + delR  
        
        delV_array.append(delV)
        delR_array.append(gamma * I + eps * delV)
        S_array.append(S)
        
        date_array_prediction.append(df['Date'].iloc[i])
        infection_array_prediction.append(alpha * E / CIR)
        
    for i in range(len(df)):
        if i == 0:
            del_confirmed[i] = int(df['Confirmed'].iloc[i]) - base_confirmed_case
        else:
            del_confirmed[i] = int(df['Confirmed'].iloc[i]) - int(df['Confirmed'].iloc[i-1])
    

    for i in range(len(df)):
        if i < 7:
            bar_del_confirmed[i] = sum(del_confirmed[:i+1]) /len(del_confirmed[:i+1])
            bar_del_infection_params[i] = sum(del_infection_params[:i+1]) /len(del_infection_params[:i+1])
            
        else:
            bar_del_confirmed[i] = sum(del_confirmed[i-6:i+1]) /len(del_confirmed[i-6:i+1])
            bar_del_infection_params[i] = sum(del_infection_params[i-6:i+1]) /len(del_infection_params[i-6:i+1])
                     
    error = np.mean((np.log(bar_del_confirmed) - np.log(bar_del_infection_params))**2)
    
    
    if prediction:
        if control == 'open loop':
            df['delV'] = delV_array
            df['delR'] = delR_array
            df['new_Cases'] = infection_array_prediction
            
            df_temp = df[['Date', 'delV', 'delR']]
            beta = frac_beta * beta
            
            delV = 200000
            date = df['Date'].iloc[-1]
            while date <= pd.to_datetime('31-12-2021',infer_datetime_format=True):
                date += datetime.timedelta(days=1)
    
                if (date >= pd.to_datetime('11-09-2021', infer_datetime_format=True)) and (date <= pd.to_datetime('31-12-2021', infer_datetime_format=True)):
                    temp_date = date - datetime.timedelta(days=180)
                    delW = int(df_temp[df_temp['Date'] == temp_date]['delR']) + eps * int(df_temp[df_temp['Date'] == temp_date]['delV'])
                    
                else:
                    delW = 0       
                
                    
                # Problem number 1
                delS = - (beta * S * I / N) -  eps * delV + delW
                delE =   (beta * S * I / N) -  alpha * E
                delI =   alpha * E - gamma * I
                delR =   gamma * I + eps * delV - delW 
                
                S = S + delS
                E = E + delE
                I = I + delI
                R = R + delR    
                
                new_row = [date, delV, delR]
                temp_len = len(df_temp)
                df_temp.loc[temp_len] = new_row
                
                S_array.append(S)
                date_array_prediction.append(date)
    
                infection_array_prediction.append(alpha * E / CIR)
    
            return date_array_prediction, infection_array_prediction, np.array(S_array)
     
        if control == 'closed loop':
            df['delV'] = delV_array
            df['delR'] = delR_array
            df['new_Cases'] = infection_array_prediction
            
            df_temp = df[['Date', 'delV', 'delR', 'new_Cases']]
            temp_beta = beta
            
            delV = 200000
            date = df['Date'].iloc[-1]
            day_count = len(df) - 1
            
            while date <= pd.to_datetime('31-12-2021', infer_datetime_format=True):
                date += datetime.timedelta(days=1)
    
                if (date >= pd.to_datetime('11-09-2021', infer_datetime_format=True)) and (date <= pd.to_datetime('31-12-2021', infer_datetime_format=True)):
                    temp_date = date - datetime.timedelta(days=180)
                    delW = int(df_temp[df_temp['Date'] == temp_date]['delR']) + eps * int(df_temp[df_temp['Date'] == temp_date]['delV'])
                    
                else:
                    delW = 0       
                if date.strftime("%A") == 'Tuesday':
                    avg_cases_prev_week = sum(df_temp['new_Cases'].iloc[day_count-7:day_count])/len(df_temp['new_Cases'].iloc[day_count-7:day_count])
                    if avg_cases_prev_week < 10000:
                        frac = 1
                    elif (avg_cases_prev_week >= 10000) and (avg_cases_prev_week < 25000):
                        frac = 2/3
                    elif (avg_cases_prev_week >= 25001) and (avg_cases_prev_week < 100000):
                        frac = 1/2
                    else:
                        frac = 1/3 
                    #print(frac)
                #print(delW)    
                delS = - (frac * temp_beta * S * I / N) -  eps * delV + delW
                delE =   (frac * temp_beta * S * I / N) -  alpha * E
                delI =   alpha * E - gamma * I
                delR =   gamma * I + eps * delV - delW 
                
                S = S + delS
                E = E + delE
                I = I + delI
                R = R + delR    
                
                new_row = [date, delV, delR, alpha * E / CIR]
                temp_len = len(df_temp)
                df_temp.loc[temp_len] = new_row
                S_array.append(S)
                
                date_array_prediction.append(date)
    
                infection_array_prediction.append(alpha * E / CIR)
                day_count+=1
            return date_array_prediction, infection_array_prediction, del_confirmed, np.array(S_array)      
        
    return error



"Does gradient descent"
def grad_descent_helper(element, params, df):
    loss = 10**6
    del_params = np.zeros(6)
    del_params[index_map[element]] = 1

    count = 1
    while loss > 0.01:
        del_loss = calculate_loss(params + del_params * step_map[element], df) - calculate_loss(params - del_params * step_map[element], df)
        params = params - (1/count) * (del_loss / 2 * step_map[element])
        loss = calculate_loss(params, df)
        count+=1
    return params

def grad_descent(params, data):
    params = grad_descent_helper(element = 'beta', params = params, df = data)
    params = grad_descent_helper(element = 'S', params = params, df = data)
    params = grad_descent_helper(element = 'E', params = params, df = data)
    params = grad_descent_helper(element = 'I', params = params, df = data)
    params = grad_descent_helper(element = 'R', params = params, df = data)
    params = grad_descent_helper(element = 'CIR', params = params, df = data)   
    return params


"predicts data for the given problem, and plots it"
def prediction(params, data, original_data):
    
    date, infec1, s_array1 = calculate_loss(params, df = data, prediction = True, frac_beta = 1, control = 'open loop')
    date, infec2, s_array2 = calculate_loss(params, df = data, prediction = True, frac_beta = 2/3, control = 'open loop')
    date, infec3, s_array3 = calculate_loss(params, df = data, prediction = True, frac_beta = 1/2, control = 'open loop')
    date, infec4, s_array4 = calculate_loss(params, df = data, prediction = True, frac_beta = 1/3, control = 'open loop')
    date, infec5, _, s_array5 = calculate_loss(params, df = data, prediction = True, frac_beta = 1/3, control = 'closed loop')
    Original_data, Original_date = get_reported_cases(original_data)
    
    plt.figure()    
    plt.plot(date, infec1, label = 'Open Loop - beta')
    plt.plot(date, infec2, label = 'Open Loop - 2*beta/3')
    plt.plot(date, infec3, label = 'Open Loop - beta/2')
    plt.plot(date, infec4, label = 'Open Loop - beta/3')
    plt.plot(date, infec5, label = 'Closed Loop')
    plt.plot(Original_date, Original_data,color = 'black', label = 'Original_Data')
    plt.xlabel('Time', fontsize = 20)
    plt.ylabel('Infections per day', fontsize = 20)
    plt.grid()
    plt.legend(prop={"size":20})
    plt.show()
    
    N = 70000000
    plt.figure()    
    plt.plot(date, s_array1/N, label = 'Open Loop - beta')
    plt.plot(date, s_array2/N, label = 'Open Loop - 2*beta/3')
    plt.plot(date, s_array3/N, label = 'Open Loop - beta/2')
    plt.plot(date, s_array4/N, label = 'Open Loop - beta/3')
    plt.plot(date, s_array5/N, label = 'Closed Loop')
    plt.xlabel('Time', fontsize = 20)
    plt.ylabel('Fraction of Susceptible population', fontsize = 20)
    plt.grid()
    plt.legend(prop={"size":20})    
    plt.show()
    return

    
"gets relevant data"
data  = pd.read_csv('COVID19_data.csv')
data= data[['Date', 'Confirmed', 'Tested', 'First Dose Administered']]
data['Date'] = pd.to_datetime(data['Date'])
base_confirmed_case = int(data[data['Date'] == '15-03-2021']['Confirmed'])
base_vaccinated_case = int(data[data['Date'] == '15-03-2021']['First Dose Administered'])
indexes_16_26 = (data['Date'] >= '16-03-2021') & (data['Date'] <= '26-04-2021')
indexes_16_20 = (data['Date'] >= '16-03-2021') & (data['Date'] <= '20-09-2021')
data_reported = data[indexes_16_20]
data = data[indexes_16_26]



"initialization_of_parameters"
"Note: values are intenionally initialised near the local minima so that code runs faster"
"Incorrect initialisation may lead to NaN error or might take a long time"
R_percentage = 30
N = 70000000
R0 = (R_percentage/100) * N
beta0 = 0.49
E0 = (0.2/100) * N
I0 = (0.1/100) * N
S0 = N - R0 - E0 - I0
CIR0 = 18.5
params = np.array([beta0, S0, E0, I0, R0, CIR0])


"helper dictionaries"
index_map = {'beta':0, 'S':1, 'E':2, 'I':3, 'R':4, 'CIR':5}
step_map = {'beta':0.01, 'S':1, 'E':1, 'I':1, 'R':1, 'CIR':0.1}

"return optimum params and loss"
optimum_params = grad_descent(params = params, data = data)
optimum_loss = calculate_loss(optimum_params, df = data, prediction = False, frac_beta = 1/3)

print('Optimum_loss is', optimum_loss)
print('Optimum parameters are mentioned below:')
print()
print('beta is ', optimum_params[0]),
print('S is ',optimum_params[1])
print('E is ',optimum_params[2])
print('I is ',optimum_params[3])
print('R is ',optimum_params[4])
print('CIR is ',optimum_params[5])


"gets the necessary plots"
prediction(params = optimum_params, data = data, original_data = data_reported)

"One can use the follwing code, if scipy module is required"
# =============================================================================
# ans = scipy.optimize.minimize(calculate_loss, params, args = data)
# print(ans)
# =============================================================================
