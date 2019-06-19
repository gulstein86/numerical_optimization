
# coding: utf-8

# ## **Fantasy Premier League Team Predictor**

# ### **Import Required Modules**

# In[1]:


import pandas as pd
import numpy as np
from pulp import *
import re
import matplotlib
import matplotlib.pyplot as plt


# ### Import List of Players

# In[2]:


df = pd.read_csv("FPL_2018_19_Wk0.csv")


# **Understand our data before start building model**

# In[3]:


# Quick view of dataset
df.head()


# In[4]:


# Check correlation between predicting factors
cov_df = df.iloc[:,5:21]
cov_df.corr()


# In[5]:


# Visualize correlation between factors
plt.figure(figsize=(20,10))
plt.matshow(cov_df.corr(), fignum = 1)
plt.xticks(range(len(cov_df.columns)), cov_df.columns)
plt.yticks(range(len(cov_df.columns)), cov_df.columns)
plt.colorbar()
plt.show()


# ### Import All Required User Defined Functions

# In[6]:


# Create player variables x1,x2,x3....
def player_variable(df):
    player_variable = []
    
    for rownum, row in df.iterrows():
        variable = str('x' + str(rownum))
        variable = pulp.LpVariable(str(variable), lowBound = 0, upBound = 1, cat= 'Integer')
        player_variable.append(variable)
                                  
    return player_variable


# In[7]:


# Create objective function
def objective_func(df,player_list,lp_prob):
    tot_pt=""
    for n,row in df.iterrows():
        for i,player in enumerate(player_list):
            if i==n:
                formula=row['Points']*player
                tot_pt+=formula
    lp_prob+=tot_pt
    return lp_prob


# In[8]:


# Create cost constraint
def cash_constraint(df,player_list,lp_prob,available_cash):
    cost=""
    formula=""
    for n,row in df.iterrows():
        for i,player in enumerate(player_list):
            if i==n:
                formula=row['Cost']*player
                cost+=formula
    lp_prob+=(cost<=available_cash),"Cash_constraint"
    return lp_prob
    


# In[9]:


# Create positions constraint
def positions(df,player_list,lp_prob,available_player,position):
    tot_player=""
    formula=""
    for n,row in df.iterrows():
        for i,player in enumerate(player_list):
            if i==n:
                if row['Position']==position:
                    formula=1*player
                    tot_player+=formula
    lp_prob+=(tot_player==available_player),position
    return lp_prob


# In[10]:


# Create assists constraint
def assists(df,player_list,lp_prob,min_assist):
    tot_assists=""
    formula=""
    for n,row in df.iterrows():
        for i,player in enumerate(player_list):
            if i==n:
                formula=row["Assists"]*player
                tot_assists+=formula
    lp_prob+=(tot_assists>=min_assist),"Assists"
    return lp_prob


# In[11]:


# Create yellow cards constraint (Number of yellow cards received)
def yellow_cards(df,player_list,lp_prob,max_yellow):
    tot_yellow=""
    formula=""
    for n,row in df.iterrows():
        for i,player in enumerate(player_list):
            if i==n:
                formula=row["Yellow_cards"]*(1)*player
                tot_yellow+=formula
    lp_prob+=(tot_yellow<=max_yellow),"Yellow_cards"
    return lp_prob


# In[12]:


# Create goals scored constraint
def goals_scored(df,player_list,lp_prob,min_goal):
    tot_goals=""
    formula=""
    for n,row in df.iterrows():
        for i,player in enumerate(player_list):
            if i==n:
                formula=row["Goals_scored"]*player
                tot_goals+=formula
    lp_prob+=(tot_goals>=min_goal),"Goals_scored"
    return lp_prob


# In[13]:


# Create number of minutes played constraint
def minutes(df,player_list,lp_prob,min_minute):
    tot_minutes=""
    formula=""
    for n,row in df.iterrows():
        for i,player in enumerate(player_list):
            if i==n:
                formula=row["Minutes"]*player
                tot_minutes+=formula
    lp_prob+=(tot_minutes>=min_minute),"Minutes Played"
    return lp_prob


# In[14]:


def find_prob(df,cash,gkp,defd,mid,fwd,assist,yellow,goals,minute):
    lp_prob=[]
    lp_prob = pulp.LpProblem('FantasyTeam', pulp.LpMaximize)
    player_list = player_variable(df)
    
    lp_prob = objective_func(df,player_list,lp_prob)
    lp_prob = cash_constraint(df,player_list,lp_prob,cash)
    lp_prob = positions(df,player_list,lp_prob,gkp,"GKP")
    lp_prob = positions(df,player_list,lp_prob,defd,"DEF")
    lp_prob = positions(df,player_list,lp_prob,mid,"MID")
    lp_prob = positions(df,player_list,lp_prob,fwd,"FWD")
    lp_prob = assists(df,player_list,lp_prob,assist)
    lp_prob = yellow_cards(df,player_list,lp_prob,yellow)
    lp_prob = goals_scored(df,player_list,lp_prob,goals)
    lp_prob = minutes(df,player_list,lp_prob,minute)
    
    return lp_prob


# In[25]:


def optimization(df, lp_prob):
    lp_prob.writeLP('Group3_FF.lp')
    
    optimization_result = lp_prob.solve()
    assert optimization_result == pulp.LpStatusOptimal
    #return optimization_result


# In[16]:


def decision(df,lp_prob):
    # append the variable x1,x2... and decison value 1,0 into dataframe
    # the index of vals in ascending order, but columns of variables are not. 
    name,val,vals=[],[],[]
    for v in lp_prob.variables():
        name.append(v.name)
        val.append(v.varValue)
    vals=pd.DataFrame({'variable': name,'value': val}) 
    
    # sort the value of column's variable in ascending order
    for n, row in vals.iterrows():
        value = re.findall(r'(\d+)', row['variable'])
        vals.loc[n, 'variable'] = int(value[0])

    df_vals = vals.sort_index(by='variable')
    
    for n,row in df.iterrows():
        for vals_n,vals_row in df_vals.iterrows():
            if n==vals_row["variable"]:
                df.loc[n,'decision']=vals_row['value']
    
    return df


# In[35]:


def diff_formation(df,cash,gkp,defd,mid,fwd,assist,yellow,goals,minute):
    lp_prob=[]
    lp_prob.clear()
    lp_prob=find_prob(df,cash,gkp,defd,mid,fwd,assist,yellow,goals,minute)
    optimization(df,lp_prob)
    final=decision(df,lp_prob)
    print("Total Cost: ",final[final['decision']==1.0].Cost.sum(),"\nTotal Points: ",final[final['decision']==1.0].Points.sum(),            "\nTotal Goals: ",final[final['decision']==1.0].Goals_scored.sum(),"\nTotal Assists: ",final[final['decision']==1.0].Assists.sum(),          "\nTotal Yellow Cards: ",final[final['decision']==1.0].Yellow_cards.sum(),"\nTotal Minutes :",final[final['decision']==1.0].Minutes.sum())
    return(final[final['decision']==1.0],lp_prob)


# ### Optimize the best team for Fantasy Premier League

# **Scenario**

# Find the best team formation with budget within 1200. One team have 2 goalkeepers, 5 defenders, 5 midfielders and 3 forwards. For entire team, assists must be at least 90, obtained not more than 20 yellow cards, scored at least 150 goals and minutes of played must be at least 735 hours (44100 minutes).

# In[26]:


import warnings
warnings.filterwarnings('ignore')


# In[36]:


final, lp_prob = diff_formation(df,1200,2,5,5,3,90,20,150,44100)


# From the output, we can have one football team of following 15 players with 1190 cash. Total assists 92, just obtained 20 yellow cards and scored 152 goals. Total minutes played for entire team is 44186 minutes.

# In[37]:


#extract out the players name and with details that we use
select=['Name', 'Team', 'Position', 'Cost','Assists','Goals_scored', 'Yellow_cards','Minutes', 'Points']
final[select].to_csv('players_output.csv',index=False, sep=',')

