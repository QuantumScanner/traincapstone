import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')


# In[90]:


# In[91]:

def main():
    st.title("FIFA Data visualization")


df = pd.read_csv('./data/international_matches.csv', parse_dates=['date'])
# df.tail()
#

# In[92]:


# df.columns


# In[93]:


# df.isnull().sum()


# # PRE-ANALYSIS
# The dataset has a lot of blank fields that need to be fixed.
# However, before modifying any field, I want to analyze the teams' qualifications on the last FIFA date (June 2022). This is important because, from these qualifications, I will create the inference dataset that enters the machine learning algorithm that predicts the World Cup matches.

# ### Top 10 FIFA Ranking
# Top 10 national teams to date FIFA June 2022.
# **ref:** https://www.fifa.com/fifa-world-ranking/men?dateId=id13603


# In[94]:


fifa_rank = df[['date', 'home_team', 'away_team', 'home_team_fifa_rank', 'away_team_fifa_rank']]
home = fifa_rank[['date', 'home_team', 'home_team_fifa_rank']].rename(
    columns={"home_team": "team", "home_team_fifa_rank": "rank"})
away = fifa_rank[['date', 'away_team', 'away_team_fifa_rank']].rename(
    columns={"away_team": "team", "away_team_fifa_rank": "rank"})
fifa_rank = pd.concat([home, away])
# Select each country latest match
fifa_rank = fifa_rank.sort_values(['team', 'date'], ascending=[True, False])
last_rank = fifa_rank
fifa_rank_top10 = fifa_rank.groupby('team').first().sort_values('rank', ascending=True)[0:10].reset_index()


# fifa_rank_top10


# ### Top 10 teams with the highest winning percentage at home and away

# In[95]:


def home_percentage(team):
    score = len(df[(df['home_team'] == team) & (df['home_team_result'] == "Win")]) / len(
        df[df['home_team'] == team]) * 100
    return round(score)


def away_percentage(team):
    score = len(df[(df['away_team'] == team) & (df['home_team_result'] == "Lose")]) / len(
        df[df['away_team'] == team]) * 100
    return round(score)


# In[96]:


fifa_rank_top10['Home_win_Per'] = np.vectorize(home_percentage)(fifa_rank_top10['team'])
fifa_rank_top10['Away_win_Per'] = np.vectorize(away_percentage)(fifa_rank_top10['team'])
fifa_rank_top10['Average_win_Per'] = round((fifa_rank_top10['Home_win_Per'] + fifa_rank_top10['Away_win_Per']) / 2)
fifa_rank_win = fifa_rank_top10.sort_values('Average_win_Per', ascending=False)
# fifa_rank_win


# ### Top 10 attacking teams in the last FIFA date

# In[97]:


fifa_offense = df[['date', 'home_team', 'away_team', 'home_team_mean_offense_score', 'away_team_mean_offense_score']]
home = fifa_offense[['date', 'home_team', 'home_team_mean_offense_score']].rename(
    columns={"home_team": "team", "home_team_mean_offense_score": "offense_score"})
away = fifa_offense[['date', 'away_team', 'away_team_mean_offense_score']].rename(
    columns={"away_team": "team", "away_team_mean_offense_score": "offense_score"})
fifa_offense = pd.concat([home, away])
fifa_offense = fifa_offense.sort_values(['date', 'team'], ascending=[False, True])
last_offense = fifa_offense
fifa_offense_top10 = fifa_offense.groupby('team').first().sort_values('offense_score', ascending=False)[
                     0:10].reset_index()
# fifa_offense_top10

import plotly.graph_objs as go
import plotly.figure_factory as ff

# In[99]:

# Display the data for the bar chart
st.write("Top 10 Attacking Teams")
st.write(fifa_offense_top10)

# Create a horizontal bar chart
fig_bar = go.Figure(data=[go.Bar(y=fifa_offense_top10['team'], x=fifa_offense_top10['offense_score'], orientation='h')])
# Update layout to include title, x-label, and y-label
fig_bar.update_layout(title='Top 10 Attacking Teams',
                      xaxis_title='Offense Score',
                      yaxis_title='Team')
st.plotly_chart(fig_bar)

# Display the data for the bar chart
# st.write("Top 10 Offense Teams")
# st.write(fifa_offense_top10)

# sns.barplot(data=fifa_offense_top10, x='offense_score', y='team', color="#7F1431")
# plt.xlabel('Offense Score', size = 20)
# plt.ylabel('Team', size = 20)
# plt.title("Top 10 Attacking teams");


# ### Top 10 Midfield teams in the last FIFA date

# In[100]:


fifa_midfield = df[['date', 'home_team', 'away_team', 'home_team_mean_midfield_score', 'away_team_mean_midfield_score']]
home = fifa_midfield[['date', 'home_team', 'home_team_mean_midfield_score']].rename(
    columns={"home_team": "team", "home_team_mean_midfield_score": "midfield_score"})
away = fifa_midfield[['date', 'away_team', 'away_team_mean_midfield_score']].rename(
    columns={"away_team": "team", "away_team_mean_midfield_score": "midfield_score"})
fifa_midfield = pd.concat([home, away])
fifa_midfield = fifa_midfield.sort_values(['date', 'team'], ascending=[False, True])
last_midfield = fifa_midfield
fifa_midfield_top10 = fifa_midfield.groupby('team').first().sort_values('midfield_score', ascending=False)[
                      0:10].reset_index()
# fifa_midfield_top10


# In[101]:

# Display the data for the bar chart
st.write("Top 10 Midfield Teams")
st.write(fifa_midfield_top10)

# Create a horizontal bar chart
fig_bar = go.Figure(
    data=[go.Bar(y=fifa_midfield_top10['team'], x=fifa_midfield_top10['midfield_score'], orientation='h')])
# Update layout to include title, x-label, and y-label
fig_bar.update_layout(title='Top 10 Midfield Teams',  # Set the title
                      xaxis_title='Midfield Score',  # Set the x-axis label
                      yaxis_title='Team')  # Set the y-axis label

# Display the bar chart
st.plotly_chart(fig_bar)

# sns.barplot(data=fifa_midfield_top10, x='midfield_score', y='team', color="#7F1431")
# plt.xlabel('Midfield Score', size = 20)
# plt.ylabel('Team', size = 20)
# plt.title("Top 10 Midfield teams");


# ### Top 10 defending teams in the last FIFA date

# In[102]:


fifa_defense = df[['date', 'home_team', 'away_team', 'home_team_mean_defense_score', 'away_team_mean_defense_score']]
home = fifa_defense[['date', 'home_team', 'home_team_mean_defense_score']].rename(
    columns={"home_team": "team", "home_team_mean_defense_score": "defense_score"})
away = fifa_defense[['date', 'away_team', 'away_team_mean_defense_score']].rename(
    columns={"away_team": "team", "away_team_mean_defense_score": "defense_score"})
fifa_defense = pd.concat([home, away])
fifa_defense = fifa_defense.sort_values(['date', 'team'], ascending=[False, True])
last_defense = fifa_defense
fifa_defense_top10 = fifa_defense.groupby('team').first().sort_values('defense_score', ascending=False)[
                     0:10].reset_index()
# fifa_defense_top10


# In[103]:

# Display the data for the bar chart
st.write("Top 10 Defensive Teams")
st.write(fifa_defense_top10)

# Create the horizontal bar chart
fig_bar = go.Figure(data=[go.Bar(y=fifa_defense_top10['team'], x=fifa_defense_top10['defense_score'], orientation='h')])

# Update layout to include title, x-label, and y-label
fig_bar.update_layout(title='Top 10 Defensive Teams',  # Set the title
                      xaxis_title='Defense Score',  # Set the x-axis label
                      yaxis_title='Team')  # Set the y-axis label

# Display the bar chart
st.plotly_chart(fig_bar)

sns.barplot(data=fifa_defense_top10, x='defense_score', y='team', color="#7F1431")
plt.xlabel('Defense Score', size=20)
plt.ylabel('Team', size=20)
plt.title("Top 10 Defense Teams")

# ### Do Home teams have any advantage?

# In[104]:


# Select all matches played at non-neutral locations
home_team_advantage = df[df['neutral_location'] == False]['home_team_result'].value_counts(normalize=True)

# # Plot
# fig, axes = plt.subplots(1, 1, figsize=(8,8))
# ax =plt.pie(home_team_advantage  ,labels = ['Win', 'Lose', 'Draw'], autopct='%.0f%%')
# plt.title('Home team match result', fontsize = 15)
# plt.show()


# As the graph shows, the home team has an advantage over the away team. This is due to factors such as the fans, the weather and the confidence of the players. For this reason, in the World Cup, those teams that sit at home will have an advantage.

# # DATA PREPARATION AND FEATURE ENGINEERING
# In this section, I will fill in the empty fields in the dataset and clean up the data for teams that did not qualify for the World Cup. Then, I will use the correlation matrix to choose the characteristics that will define the training dataset of the Machine Learning model. Finally, I will use the ratings of the teams in their last match to define the "Last Team Scores" dataset (i.e., the dataset that I will use to predict the World Cup matches).

# ### Analyze and fill na's

# In[105]:

#
# df.isnull().sum()


# In[106]:


# We can fill mean for na's in goal_keeper_score
df[df['home_team'] == "Brazil"]['home_team_goalkeeper_score'].describe()

# In[107]:


df['home_team_goalkeeper_score'] = round(
    df.groupby("home_team")["home_team_goalkeeper_score"].transform(lambda x: x.fillna(x.mean())))
df['away_team_goalkeeper_score'] = round(
    df.groupby("away_team")["away_team_goalkeeper_score"].transform(lambda x: x.fillna(x.mean())))

# In[108]:


# We can fill mean for na's in defense score
df[df['away_team'] == "Uruguay"]['home_team_mean_defense_score'].describe()

# In[65]:


df['home_team_mean_defense_score'] = round(
    df.groupby('home_team')['home_team_mean_defense_score'].transform(lambda x: x.fillna(x.mean())))
df['away_team_mean_defense_score'] = round(
    df.groupby('away_team')['away_team_mean_defense_score'].transform(lambda x: x.fillna(x.mean())))

# In[109]:


# We can fill mean for na's in offense score
df[df['away_team'] == "Uruguay"]['home_team_mean_offense_score'].describe()

# In[67]:


df['home_team_mean_offense_score'] = round(
    df.groupby('home_team')['home_team_mean_offense_score'].transform(lambda x: x.fillna(x.mean())))
df['away_team_mean_offense_score'] = round(
    df.groupby('away_team')['away_team_mean_offense_score'].transform(lambda x: x.fillna(x.mean())))

# In[110]:


# We can fill mean for na's in midfield score
df[df['away_team'] == "Uruguay"]['home_team_mean_midfield_score'].describe()

# In[111]:


df['home_team_mean_midfield_score'] = round(
    df.groupby('home_team')['home_team_mean_midfield_score'].transform(lambda x: x.fillna(x.mean())))
df['away_team_mean_midfield_score'] = round(
    df.groupby('away_team')['away_team_mean_midfield_score'].transform(lambda x: x.fillna(x.mean())))

# In[112]:


df.isnull().sum()

# In[113]:


# Teams are not available in FIFA game itself, so they are not less than average performing teams, so giving a average score of 50 for all.
df.fillna(50, inplace=True)

# ### Filter the teams participating in QATAR - World cup 2022

# In[115]:


list_2022 = ['Qatar', 'Germany', 'Denmark', 'Brazil', 'France', 'Belgium', 'Croatia', 'Spain', 'Serbia', 'England',
             'Switzerland', 'Netherlands', 'Argentina', 'IR Iran', 'Korea Republic', 'Japan', 'Saudi Arabia', 'Ecuador',
             'Uruguay', 'Canada', 'Ghana', 'Senegal', 'Portugal', 'Poland', 'Tunisia', 'Morocco', 'Cameroon', 'USA',
             'Mexico', 'Wales', 'Australia', 'Costa Rica']
final_df = df[(df["home_team"].apply(lambda x: x in list_2022)) | (df["away_team"].apply(lambda x: x in list_2022))]

# **Top 10 teams in QATAR 2022**

# In[116]:


rank = final_df[['date', 'home_team', 'away_team', 'home_team_fifa_rank', 'away_team_fifa_rank']]
home = rank[['date', 'home_team', 'home_team_fifa_rank']].rename(
    columns={"home_team": "team", "home_team_fifa_rank": "rank"})
away = rank[['date', 'away_team', 'away_team_fifa_rank']].rename(
    columns={"away_team": "team", "away_team_fifa_rank": "rank"})
rank = pd.concat([home, away])

# Select each country latest match
rank = rank.sort_values(['team', 'date'], ascending=[True, False])
rank_top10 = rank.groupby('team').first().sort_values('rank', ascending=True).reset_index()
rank_top10 = rank_top10[(rank_top10["team"].apply(lambda x: x in list_2022))][0:10]

st.write("Top 10 Countries by Rank - Latest Match")
rank_top10

# # Create a scatter plot
# fig_scatter = go.Figure(data=go.Scatter(x=rank_top10['team'], y=rank_top10['rank'], mode='markers', marker=dict(color='lightskyblue', size=12)))
#
# # Update layout to include title and labels
# fig_scatter.update_layout(title='Top 10 Countries by Rank - Latest Match',
#                           xaxis_title='Country',
#                           yaxis_title='Rank')
#
# # Display the scatter plot
# st.plotly_chart(fig_scatter)

# **Top 10 teams with the highest winning percentage in QATAR 2022**

# In[117]:


rank_top10['Home_win_Per'] = np.vectorize(home_percentage)(rank_top10['team'])
rank_top10['Away_win_Per'] = np.vectorize(away_percentage)(rank_top10['team'])
rank_top10['Average_win_Per'] = round((rank_top10['Home_win_Per'] + rank_top10['Away_win_Per']) / 2)
rank_top10_Win = rank_top10.sort_values('Average_win_Per', ascending=False)

# st.write("Top 10 Countries by Rank - Latest Match")
# rank_top10_Win


# In[118]:

# Display the data for the bar chart
st.write("Top 10 Average Win Per game Teams")
st.write(rank_top10_Win)

# Create a horizontal bar chart
# Create a horizontal bar chart
fig_bar = go.Figure(data=[go.Bar(y=rank_top10_Win['team'], x=rank_top10_Win['Average_win_Per'], orientation='h')])

# Update layout to include title and labels
fig_bar.update_layout(title='Top 10 Countries by Average Win Percentage',
                      xaxis_title='Average Win Percentage',
                      yaxis_title='Country')

# Display the horizontal bar chart
st.plotly_chart(fig_bar)

# sns.barplot(data=rank_top10_Win,x='Average_win_Per',y='team',color="#7F1431")
# plt.xticks()
# plt.xlabel('Win Average', size = 20)
# plt.ylabel('Team', size = 20)
# plt.title('Top 10 QATAR 2022 teams with the highest winning percentage')

#
# # ### Correlation Matrix
#
# # In[124]:
#
#
# final_df['home_team_result'].values
# # for index, value in final_df['home_team_result'].items():
# #     print(f"Row {index}: {value}")
#
#
# # In[125]:
#
#
# team_result_df = final_df
# # for index, value in team_result_df['home_team_result'].items():
# #     print(f"Row {index}: {value}")
#
#
# # In[151]:
#
#
# # Mapping numeric values for home_team_result to find the correleations
# final_df['home_team_result'] = final_df['home_team_result'].map({'Win':1, 'Draw':2, 'Lose':0})
#
#
# # In[145]:
#
#
#
#
#
# # In[150]:
#
#
# final_df['home_team_result'].head(1)
#
#
# # In[152]:
#
#
# final_df['home_team_result'] = pd.to_numeric(final_df['home_team_result'], errors='coerce')
#
#
# # In[155]:
#
#
# # df.head()
#
#
# # In[156]:
#
#
# # final_df.head()
#
#
# # In[157]:
#
#
# numerical_df = final_df.select_dtypes(include=['number'])
#
#
# # In[158]:
#
#
# numerical_df.corr()['home_team_result'].sort_values(ascending=False)
#
#
# # In[153]:
#
#
# # final_df.corr()['home_team_result'].sort_values(ascending=False)
#
#
# # Dropping unnecessary colums.
#
# # In[ ]:
#
#
# #Dropping unnecessary colums
# final_df = final_df.drop(['date', 'home_team_continent', 'away_team_continent', 'home_team_total_fifa_points', 'away_team_total_fifa_points', 'home_team_score', 'away_team_score', 'tournament', 'city', 'country', 'neutral_location', 'shoot_out'],axis=1)
#
#
# # In[ ]:
#
#
# # final_df.columns
#
#
# # In[ ]:
#
#
# # Change column names
# final_df.rename(columns={"home_team":"Team1", "away_team":"Team2", "home_team_fifa_rank":"Team1_FIFA_RANK",
#                          "away_team_fifa_rank":"Team2_FIFA_RANK", "home_team_result":"Team1_Result", "home_team_goalkeeper_score":"Team1_Goalkeeper_Score",
#                         "away_team_goalkeeper_score":"Team2_Goalkeeper_Score", "home_team_mean_defense_score":"Team1_Defense",
#                         "home_team_mean_offense_score":"Team1_Offense", "home_team_mean_midfield_score":"Team1_Midfield",
#                         "away_team_mean_defense_score":"Team2_Defense", "away_team_mean_offense_score":"Team2_Offense",
#                         "away_team_mean_midfield_score":"Team2_Midfield"}, inplace=True)
#
#
# # In[ ]:
#
#
# plt.figure(figsize=(10, 4), dpi=200)
# sns.heatmap(final_df.corr(), annot=True)
#
#
# # In[ ]:
#
#
# # final_df.info()
#
#
# # In[ ]:
#
#
# # final_df
#
#
# # Exporting the training dataset.
#
# # In[ ]:
#
#
# # final_df.to_csv("./data/training.csv", index = False)
#
#
# # ### Creating "Last Team Scores" dataset
# # This dataset contains the qualifications of each team on the previous FIFA date and will be used to predict the World Cup matches.
#
# # In[ ]:
#
#
# last_goalkeeper = df[['date', 'home_team', 'away_team', 'home_team_goalkeeper_score', 'away_team_goalkeeper_score']]
# home = last_goalkeeper[['date', 'home_team', 'home_team_goalkeeper_score']].rename(columns={"home_team":"team", "home_team_goalkeeper_score":"goalkeeper_score"})
# away = last_goalkeeper[['date', 'away_team', 'away_team_goalkeeper_score']].rename(columns={"away_team":"team", "away_team_goalkeeper_score":"goalkeeper_score"})
# last_goalkeeper = pd.concat([home,away])
#
# last_goalkeeper = last_goalkeeper.sort_values(['date', 'team'],ascending=[False, True])
#
# list_2022 = ['Qatar', 'Germany', 'Denmark', 'Brazil', 'France', 'Belgium', 'Croatia', 'Spain', 'Serbia', 'England', 'Switzerland', 'Netherlands', 'Argentina', 'IR Iran', 'Korea Republic', 'Japan', 'Saudi Arabia', 'Ecuador', 'Uruguay', 'Canada', 'Ghana', 'Senegal', 'Portugal', 'Poland', 'Tunisia', 'Morocco', 'Cameroon', 'USA', 'Mexico', 'Wales', 'Australia', 'Costa Rica']
#
# rank_qatar = last_rank[(last_rank["team"].apply(lambda x: x in list_2022))]
# rank_qatar = rank_qatar.groupby('team').first().reset_index()
# goal_qatar = last_goalkeeper[(last_goalkeeper["team"].apply(lambda x: x in list_2022))]
# goal_qatar = goal_qatar.groupby('team').first().reset_index()
# goal_qatar = goal_qatar.drop(['date'], axis = 1)
# off_qatar = last_offense[(last_offense["team"].apply(lambda x: x in list_2022))]
# off_qatar = off_qatar.groupby('team').first().reset_index()
# off_qatar = off_qatar.drop(['date'], axis = 1)
# mid_qatar = last_midfield[(last_midfield["team"].apply(lambda x: x in list_2022))]
# mid_qatar = mid_qatar.groupby('team').first().reset_index()
# mid_qatar = mid_qatar.drop(['date'], axis = 1)
# def_qatar = last_defense[(last_defense["team"].apply(lambda x: x in list_2022))]
# def_qatar = def_qatar.groupby('team').first().reset_index()
# def_qatar = def_qatar.drop(['date'], axis = 1)
#
# qatar = pd.merge(rank_qatar, goal_qatar, on = 'team')
# qatar = pd.merge(qatar, def_qatar, on ='team')
# qatar = pd.merge(qatar, off_qatar, on ='team')
# qatar = pd.merge(qatar, mid_qatar, on ='team')
#
# qatar['goalkeeper_score'] = round(qatar["goalkeeper_score"].transform(lambda x: x.fillna(x.mean())))
# qatar['offense_score'] = round(qatar["offense_score"].transform(lambda x: x.fillna(x.mean())))
# qatar['midfield_score'] = round(qatar["midfield_score"].transform(lambda x: x.fillna(x.mean())))
# qatar['defense_score'] = round(qatar["defense_score"].transform(lambda x: x.fillna(x.mean())))
# # qatar.head(5)
#
#
# # Exporting the "Last Team Scores" dataset.
#
# # In[ ]:
#

# qatar.to_csv("/content/drive/MyDrive/data/last_team_scores.csv", index = False)

if __name__ == "__main__":
    main()
