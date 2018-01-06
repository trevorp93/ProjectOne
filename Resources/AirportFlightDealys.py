
# coding: utf-8

# In[2]:


import unicodecsv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from scipy import stats
from scipy.stats import norm

pd.options.display.float_format = '{:.0f}'.format


# In[3]:


flights = pd.read_csv("73515083_102017_2041_airline_delay_causes.csv")

len(flights)


# In[4]:


flights.columns


# In[5]:


flights.info()

flights["carrier"] = flights["carrier"].astype("category")
flights["carrier_name"] = flights["carrier_name"].astype("category")
flights["airport"] = flights["airport"].astype("category")
flights["airport_name"] = flights["airport_name"].astype("category")


# In[6]:


#flights.drop([' arr_delay'], axis=1, inplace=True)
#flights.drop([' carrier_delay'], axis=1, inplace=True)
#flights.drop(['weather_delay'], axis=1, inplace=True)
#flights.drop(['nas_delay'], axis=1, inplace=True)
#flights.drop(['security_delay'], axis=1, inplace=True)
#flights.drop(['late_aircraft_delay'], axis=1, inplace=True)


# In[7]:


flights.info()


# In[8]:


flights = flights.rename(columns={' month': 'Month', 'arr_del15': 'Arrival_Delay', 'carrier_ct':'Carrier_Delay', ' weather_ct': 'Weather_Delay',
                        'nas_ct': 'NAS_Delay', 'security_ct': "Security_Delays", 'late_aircraft_ct': 'Late_Aircraft_Delay', 
                        'arr_cancelled': 'Cancelled', 'arr_diverted': 'Diverted',
                        'year': 'Year'
                       }
              )


# In[9]:


flights = flights.rename(columns={'Security_Delays' :"Security_Delay"})


# In[10]:


flights.to_csv('reformed_data.csv', encoding='utf-8', index=False)

