#importing general objects
import pandas as pd
import plotly.express as px
import numpy as np
import streamlit as st

#TODO: reading in the dataset


#Some basic commands in streamlit -- you can find an amazing cheat sheet here: https://docs.streamlit.io/library/cheatsheet
st.title('Determining Income Level EDA')
st.write('Politicians often value individual income because of its influence on the country\'s GDP and per capita. Through the annual census, different demographic features can be collected for further exploration by computer scientists. This analysis explores features, like current working status, education, sex, race, and native country; it identifies which factors are most pertinent towards gauging income. Especially during inflationary and deflationary periods, these correlations become much more important.')
st.write('To clean the data, missing values were removed, the data was shuffled, the \'income binary\' column changed from categorical to numerical, the data removed bias (reducing the disparity bewteen categories), and less useful columns were dropped based on feature selection algorithms.')
st.markdown("""---""")
#generate random data for my example dataframe -- howto: https://stackoverflow.com/questions/32752292/how-to-create-a-dataframe-of-random-integers-with-pandas
df = pd.read_csv('clean_df_large.csv')
del df[df.columns[0]]
del df[df.columns[0]]
del df[df.columns[2]]
df = df.sample(frac = 1)
df = df.reset_index()
del df[df.columns[0]]
df = df[~(df == '?').any(axis=1)]

#show off a bit of your data. 
st.header('Census Income Data')
# col1, col2 = st.columns(2) #here is how you can use columns in streamlit. 
st.markdown("For clarification, values with ? mean that it was not in the options provided.")
st.dataframe(df.head())
df[" work class"].replace(df[" work class"].unique(), range(len(df[" work class"].unique())), inplace=True)
df[" education-num"].replace(df[" education-num"].unique(), range(len(df[" education-num"].unique())), inplace=True)
df[" marital-status"].replace(df[" marital-status"].unique(), range(len(df[" marital-status"].unique())), inplace=True)
df[" occupation"].replace(df[" occupation"].unique(), range(len(df[" occupation"].unique())), inplace=True)
df[" relationship"].replace(df[" relationship"].unique(), range(len(df[" relationship"].unique())), inplace=True)
df[" race"].replace(df[" race"].unique(), range(len(df[" race"].unique())), inplace=True)
df[" sex"].replace(df[" sex"].unique(), range(len(df[" sex"].unique())), inplace=True)
df[" native-country"].replace(df[" native-country"].unique(), range(len(df[" native-country"].unique())), inplace=True)

st.markdown('This dataframe has over 250K rows (reduced from 6B rows), with each having 13 features and one output column. This data is representative of the global population and unbiased due to its method of collection.') #you can add multiple items to each column.
st.markdown('To see the raw unparsed data, reference [this dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income). This data will be used for binary classification.')
df["Age"] = (df["Age"]-df["Age"].mean())/df["Age"].std()
df[" education-num"] = (df[" education-num"]-df[" education-num"].mean())/df[" education-num"].std()
df[" capital-gain"] = (df[" capital-gain"]-df[" capital-gain"].mean())/df[" capital-gain"].std()
df[" capital-loss"] = (df[" capital-loss"]-df[" capital-loss"].mean())/df[" capital-loss"].std()
df[" hours-per-week"] = (df[" hours-per-week"]-df[" hours-per-week"].mean())/df[" hours-per-week"].std()

st.dataframe(df.head())
st.markdown("To get a better idea of the final data, here's the data with numerical columns normalized and categorical columns into numerical labels. Income-binary has two labels: 0 (less than 50K) and 1 (greater than 50K).")
st.markdown("""---""")

st.header('Correlation Analysis of Income')
st.plotly_chart(px.imshow(df.corr()))
st.markdown("Looking at the correlation heatmap, we see that no feature has an extremely strong correlation with income. As expected, age and hours per week worked play a positive role in income. Meanwhile, marital status and relationship have extreme negative correlations with income.")
st.markdown("""---""")

df2 = pd.read_csv('clean_df_large.csv')
del df2[df2.columns[0]]
del df2[df2.columns[0]]
del df2[df2.columns[2]]
df2 = df2.sample(frac = 1)
df2 = df2.reset_index()
del df2[df2.columns[0]]
df2 = df2[~(df2 == '?').any(axis=1)]

st.header("Histogram Analysis (Categorical Variables)")
col1, col2 = st.columns(2)
col1.plotly_chart(px.histogram(df2, 
                 x=" income-binary",
                 color=" race",
                 hover_data=df2.columns,
                 title="Distribution of Income",
                 barmode="group"))
col2.plotly_chart(px.histogram(df2, 
                 x=" income-binary",
                 color=" sex",
                 hover_data=df2.columns,
                 title="Distribution of Income ",
                 barmode="group"))
st.markdown("Looking at the two most pertinent features from the correlation matrix, we see that white males take a proportionate amount of histogram. In the race histogram, it's less clear which group is disproportionately affected, since the bar graph doesn't provide much information; however, in the second graph, we see that females have a much lower percentage in the 1 (greater than 50K) category.")
col1, col2 = st.columns(2)
col1.plotly_chart(px.histogram(df2, 
                 x=" income-binary",
                 color=" marital-status",
                 hover_data=df2.columns,
                 title="Distribution of Income",
                 barmode="group"))
col2.plotly_chart(px.histogram(df2, 
                 x=" income-binary",
                 color=" relationship",
                 hover_data=df2.columns,
                 title="Distribution of Income ",
                 barmode="group"))
st.markdown("Looking at the two most negative correlations from the correlation matrix, we see much stronger results. Individuals that are never-married, not-in-family, an own child, widowed, or divorced are much more likely to fall into the 0 (less than 50K) category. Married spouses have more even distributions of income.")
col1, col2 = st.columns(2)
col1.plotly_chart(px.histogram(df2, 
                 x=" occupation",
                 color=" income-binary",
                 hover_data=df2.columns,
                 title="Distribution of Income",
                 barmode="group"))
df2.drop(df2[df2[' native-country'] == " United-States"].index, inplace = True)
df2.drop(df2[df2[' native-country'] == " ?"].index, inplace = True)
df2.drop(df2[df2[' native-country'] == " Mexico"].index, inplace = True)
st.markdown("Because there are much more categories for occupation and native country, the histogram changed to place the categories on the x-axis. For occupation, we see that managerial and professional roles have even splits while machine inspectors and clerical roles have more falling in the 0 (less than 50K) category.")
st.markdown("The data was extremely biased towards US and Mexico, not providing an accurate representation, so those countries were removed. We see Latin American countries like El Salvador and Phillipines have more people in the 0 (less than 50K) category while India and Taiwan have even splits.")
col2.plotly_chart(px.histogram(df2, 
                 x=" native-country",
                 color=" income-binary",
                 hover_data=df2.columns,
                 title="Distribution of Income ",
                 barmode="group"))
st.markdown("""---""")

st.header("Box and Whisker Analysis (Continuous Variables)")
col1, col2 = st.columns(2)
col1.plotly_chart(px.box(df2,y=" hours-per-week",x=" income-binary",title=f"Distribution of Hours Worked Per Week"))
col2.plotly_chart(px.box(df2,y=" education-num",x=" income-binary",title=f"Distribution of Years of Education"))
st.markdown("To plot the continuous variables, a box and whisker plot strategy was taken. We clearly see that working more hours per week and having more education leads to higher income.")
st.markdown("In terms of outliers, most individuals work 40 hours a week, with a few outliers in 100 hrs/wk. Also, for education, the average is from 9-12 years, with some outlier individuals having little to no education.")
st.markdown("""---""")

#Always good to section out your code for readability.
st.header('Conclusions')
st.markdown("Here were some of the findings found from the income dataset:")
st.markdown("- No features had strong correlations with income; however, some significant correlations were age (positive), hours worked per week (positive), marital status (negative), and relationship (negative).")
st.markdown("- The data was disproportionally skewed towards whites, males, and people living in the United States.")
st.markdown("- Females, non-married individuals, and clerical inspectors were found to have lower incomes.")
st.markdown("- People in Latin American countries, with lower working hours, or less years of education were found to have lower incomes. This was done through histogram and box/whisker plot analysis.")
st.markdown("\n")

st.markdown('Some additional steps that can be taken (with machine learning):')
st.markdown('- Teams could use decision trees (or random forest trees) on this binary classification dataset to definitively see which factors are most susceptible to income level.')
st.markdown('- Teams can incorporate neural networks and/or logistic regression models to identify whether an individual falls below or above 50K given a test dataset of different factors.')
