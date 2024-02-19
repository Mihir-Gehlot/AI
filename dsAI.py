import os
from apikey import apikey

import streamlit as st
import pandas as pd

from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv




#OpenAI key
os.environ['OPENAI_API_KEY'] = apikey
load_dotenv(find_dotenv())


st.title(' AI Assistant for Data Science')
st.header('Created by Mihir Gehlot')


with st.sidebar:
    st.write(" Data Science AI")
    st.caption("I'm Mihir Gehlot, navigating the intersection of data and artificial intelligence in my latest project. Unraveling insights, This AI transform raw data into informed decisions for a smarter future.")
  
    st.divider()
    st.caption ("<p style='text-align:center'> Cooked by Mihir</p>", unsafe_allow_html=True)

#Button are true only for a moment when they are clikced they turn back to false

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let get started", on_click=clicked, args=[1])

if st.session_state.clicked[1]:
    st.header("Exploratory Data Analysis Part")
    st.subheader("Solution")

    user_csv = st.file_uploader(("Upload your CSV file"), type="csv")

    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)

        # llm model
        llm = OpenAI(temperature = 0)

        @st.cache_data  # @st.cache_data can be written as steps_eda = st.cache_data(steps_eda)
        def steps_eda():
            llm('What are the steps of EDA')
            return steps_eda


        # df agent helps in questioning dataframe
        # verbose is true, output is generated even if there are no failures.

        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose = True)        
        
        
        @st.cache_data
        def function_agent ():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head())
            st.write("**Data Cleaning**")
            columns_df = pandas_agent.run("What are the meaning of the columns?")
            st.write(columns_df)
            missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
            st.write(missing_values)
            duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
            st.write(duplicates)
            st.write("**Data Summarisation**")
            st.write(df.describe())
            correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
            st.write(correlation_analysis)
            outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
            st.write(outliers)
            new_features = pandas_agent.run("What new features would be interesting to create?.")
            st.write(new_features)
            return
        
        @st.cache_data
        def function_question_variable():
            st.line_chart(df, y = [user_question_variable])
            summary_statistics = pandas_agent.run(f"Give me summary of {user_question_variable}")
            st.write(summary_statistics)
            normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable}")
            st.write(normality)
            outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
            st.write(outliers)
            trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
            st.write(trends)
            missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
            st.write(missing_values)
            return
        
        @st.cache_data
        def function_question_dataframe():
            dataframe_info = pandas_agent.run(user_question_dataframe)
            st.write(dataframe_info)
            return

        with st.sidebar:
            with st.expander("What are the steps of EDA"):
                st.write(steps_eda())


        function_agent()
        st.subheader('Variable of study')
        user_question_variable = st.text_input('What Variable you want to know more?')
        if user_question_variable is not None and user_question_variable !=("","no","No"):
            function_question_variable()
        

            st.subheader('Further Study')

        if user_question_variable:
            user_question_dataframe = st.text_input('Is there anything else you want to knwo about the dataframe?')
            if user_question_dataframe is not None and user_question_variable !=("","no","No"):
                function_question_dataframe()
            if user_question_variable in ("no","No"):
                st.write("")

