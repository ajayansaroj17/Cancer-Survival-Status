import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config("Cancer Suvival Status")
st.title("Cancer Survival Status")
st.markdown("---")
st.header("About Dataset")
st.markdown("The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.")

# read csv
df=pd.read_csv('./haberman.csv',names=["age","operation_Year","axil_nodes","survival_status"])
st.dataframe(df)
# convert categorical to numeral
df["survival_status"] = df["survival_status"].map({1:"Yes", 2:"No"})
df.head(10)

st.markdown(f"Number of points : {df.shape[0]}")
st.markdown(f"Number of features : {df.shape[1]}")
# m,n = df.shape  --> or we can find number of points and features through this also
st.markdown("---")
st.subheader("**Objective :**")
st.markdown("Surival status of person over 5 year after his/her operation")

# sns
sns.set_style('whitegrid')
fig = sns.FacetGrid(df, hue="survival_status", height=4).set(title='Suvival scatter plot') \
 .map(plt.scatter, "axil_nodes", "age") \
 .add_legend();
st.pyplot(fig)

#
st.subheader('Univariate analysis - Plot PDF, CDF, Boxplot, Voilin plots')
fig1 = sns.FacetGrid(df,hue="survival_status",height = 5).set(title='PDF for age')\
 .map(sns.histplot,"age",kde=True,stat="density", linewidth=0)\
 . add_legend();
st.pyplot(fig1)

st.markdown('> By observing the overlapping we can see that people whose age is in the range 30–40 are more likely to survive, and 40–60 are less likely to survive. While people whose age is in the range 60–75 have equal chances of surviving and not surviving.Taking the results from other parameters as well')
st.markdown("---")

#
fig2 = sns.FacetGrid(df,hue="survival_status",height = 5).set(title='PDF for Operation year')\
 .map(sns.histplot,"operation_Year",kde=True,stat="density", linewidth=0)\
 . add_legend();
st.pyplot(fig2)
st.markdown('It is observed that in the years 1958 and 1965 there were more unsuccessful operations.')
st.markdown("---")

#
fig3 = sns.FacetGrid(df,hue="survival_status",height = 5).set(title='PDF for Axil_node')\
 .map(sns.histplot,"axil_nodes",kde=True,stat="density", linewidth=0)\
 . add_legend();
st.pyplot(fig3)
st.markdown('A patient with 25 or more nodes have very few chances of his/her survival')
st.markdown("---")

#
status_yes = df[df["survival_status"]=="Yes"]
status_no = df[df["survival_status"]=="No"]
#status_yes dataframe stores all the records where status is yes
#status_no dataframe stores all the records where status is no
counts1, bin_edges1 = np.histogram(status_yes['axil_nodes'], bins=10, density = True)
pdf1 = counts1/(sum(counts1))
print(pdf1);
print(bin_edges1)
cdf1 = np.cumsum(pdf1)
plt.plot(bin_edges1[1:], pdf1)
plt.plot(bin_edges1[1:], cdf1, label = 'Yes')
plt.xlabel('Nodes')
print("\n *****     ****     *****   ******   **** ****   ****  ** \n")
counts2, bin_edges2 = np.histogram(status_no['axil_nodes'], bins=10, density = True)
pdf2 = counts2/(sum(counts2))
print(pdf2);
print(bin_edges2)
cdf2 = np.cumsum(pdf2)
plt.plot(bin_edges2[1:], pdf2)
plt.plot(bin_edges2[1:], cdf2, label = 'No')
plt.xlabel('Nodes')
plt.legend()
plt.title("CDF for Survival")
st.pyplot(plt)

st.markdown('Orange line shows there is a 83% chance of long survival if number of axil_nodes detected are < 5 \n There are nearly 55% of people who have nodes less than 5 and there are nearly 100% of people in short survival if nodes are > 40')
st.markdown('---')
# Following plot show survival status based on axil_nodes

# Survival years is the class attribute. Classification of a class attribute over whole  instances 306 could be seen as 81 in the class(2) i.e., 81 instances are of patients who survived over 5 years after operation and 225 instances are of patients who could not survive over 5 years after the operation
st.subheader('Box plot analysis')
sns.boxplot(x='survival_status',y='age',data=df).set(title='Survival based on age')
plt.show()
st.pyplot(plt)
sns.boxplot(x='survival_status',y='operation_Year',data=df).set(title='Survival based on operation year')
plt.show()
st.pyplot(plt)
sns.boxplot(x='survival_status',y='axil_nodes',data=df).set(title='Survival based axil_nodes')
plt.show()
st.pyplot(plt)
st.markdown('From axil_nodes box plot we say that a large outlier is present 25th percentile and 50th percentile are nearly same for Long survive and threshold for it is 0 to 7. \n So,if nodes between 0–7 have chances of error as short survival plot is also lies in it. That is 50% error for Short survival status')
st.markdown('---')

#
st.subheader("Violon plot analysis")
sns.violinplot(x='survival_status',y='age',data=df,height = 10).set(title='Survival based on age')
plt.show()
st.pyplot(plt)
sns.violinplot(x='survival_status',y='operation_Year',data=df,height = 10).set(title='Survival based on operation year')
plt.show()
st.pyplot(plt)
sns.violinplot(x='survival_status',y='axil_nodes',data=df,height = 10).set(title='Survival based on axil_nodes')
plt.show()
st.pyplot(plt)
st.markdown(" More the number of nodes less is the survival chances \n Comparatively there are more people who got operated in the year 1965 did not survive for more than 5 years.")
st.markdown('---')
st.subheader('Bivariate analysis - Plot 2D Scatter plots and Pair plots')
# scatterplot
sns.set_style("whitegrid")
sns.FacetGrid(df, hue = "survival_status" , height = 6).set(title='Scatter plot based on age')\
 .map(plt.scatter,"age","operation_Year")\
 .add_legend()
plt.show()
st.pyplot(plt)
st.markdown('Patients with no nodes have more chances of survival irrespective of their age \n There are very few patients with nodes more than 25')
st.markdown('---')
# pairplot
st.subheader('Pair Plot')
sns.set_style("whitegrid")
sns.pairplot(df, hue="survival_status", height = 5)
plt.show()
st.pyplot(plt)

st.markdown('---')
st.subheader('Conclusion')
st.markdown('1. Along with age and operational_year even features like axil_nodes are needed to classify survival_status of person \n 2. We can say that more the number of nodes lesser is chance of survival for patient \n 3. As dataset is imbalanced so classifying the survival status of a new patient based on the given features is a difficult task')
st.subheader('References:')
st.write("[Reference 1](https://towardsdatascience.com/exploratory-data-analysis-habermans-cancer-survival-dataset-c511255d62cb)")
st.write("[Reference_2](https://towardsdatascience.com/will-habermans-survival-data-set-make-you-diagnose-cancer-8f40b3449673)")