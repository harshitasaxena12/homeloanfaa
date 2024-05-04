import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load the data (same as in the notebook)
def load_data():
    df = pd.read_csv('Loan_Data.csv')
    return df

# call the method to load the data and assign it to dataframe
df = load_data()
mstatus = df['Loan_Status'].value_counts()
gen = df['Gender'].value_counts()
property_area = df['Property_Area'].value_counts()


# method to show the explore page 
def show_explore_page():
    
    # set the title of the page
    st.title("Housing Home Loans Analysis")
    
    # create tabs to load the anaysis charts seprately
    distribution_tab, multivariate_tab = st.tabs(["Distribution", "Raw Data"])
    
    #using the "with" display charts under this tab
    with distribution_tab:
        
        # create two columns to load the data side by side
        cols = st.columns(2)
        
        # load the charts using seaborn in the first column (left side)
        with cols[0]:
            
            st.write("## Distribution of Loan Amount ##")
            st.line_chart(data=df['LoanAmount'] )
            
            # fig1 = plt.figure(figsize=(10, 5))
            # sns.histplot(df['LoanAmount'], bins = 20)
            # plt.title('Distribution of Loan amount', fontdict={'fontname' : 'Monospace', 'fontsize' : 30, 'fontweight' : 'bold'})
            # plt.xlabel('Load amount', fontdict = {'fontname' : 'Monospace', 'fontsize' : 30})
            # plt.ylabel('Number of people', fontdict = {'fontname' : 'Monospace', 'fontsize' : 30})
            # st.pyplot(fig1)
            

            
            st.write("## Distribution of Loan status ##")
            st.bar_chart(data=mstatus )
            
            # fig2 = plt.figure(figsize=(10, 5))
            # sns.barplot(mstatus.index, mstatus.values)
            # plt.title('Distribution of Loan status', fontdict = {'fontname' : 'Monospace', 'fontsize' : 20, 'fontweight' : 'bold'})
            # plt.xlabel('Loan status', fontdict = {'fontname' : 'Monospace', 'fontsize' : 15})
            # plt.ylabel('Number of people', fontdict = {'fontname' : 'Monospace', 'fontsize' : 15})
            # st.pyplot(fig2)
            
        # load the charts using seaborn in the second column (right side)
        with cols[1]:
            
            st.write("## Distribution of Gender ##")
            st.bar_chart(data=gen )

            # fig3 = plt.figure(figsize=(10, 5))
            # sns.barplot(gen.index, gen.values)
            # plt.title('Distribution of Gender', fontdict = {'fontname' : 'Monospace', 'fontsize' : 20, 'fontweight' : 'bold'})
            # plt.xlabel('Gender', fontdict = {'fontname' : 'Monospace', 'fontsize' : 30})
            # plt.ylabel('Number of people', fontdict = {'fontname' : 'Monospace', 'fontsize' : 30})
            # st.pyplot(fig3)
            
            st.write("## Property area distribution ##")
            st.bar_chart(data=property_area )
            
            # fig4 = plt.figure(figsize=(10, 5))
            # plt.pie(property_area.values, shadow=True,labels=property_area.index, explode = (0.05, 0.05, 0.05),autopct = '%1.1f%%', startangle=30,)
            # plt.title('Property area distribution', fontdict = {'fontname' : 'Monospace', 'fontsize' : 30, 'fontweight' : 'bold'})
            # plt.legend()
            # plt.legend(prop={'size':20})
            # plt.axis('equal')
            # st.pyplot(fig4)
    
    # second tab to display the data
    with multivariate_tab:

        st.dataframe(df)
