
import streamlit as st
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
import os
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Fake News Dector", page_icon="🇺🇸", layout="wide")
# Add a sidebar with information
st.sidebar.title("About")
st.sidebar.info(
    "This app uses AI agents to analyze news articles"
    "and predict if they contain false or misleading information. Enter a news article "
    "in the main panel to get started."
)

# Set up the Streamlit app
st.title("Fake News Dector")

# Input for news article
news_input = st.text_area("Enter the news article to analyze:")



# Create the search tool
search_tool = DuckDuckGoSearchRun()


# Agent
researcher = Agent(
    role="Research Analyst",
    goal="Find relevant information about the news article",
    backstory="You are an expert in political news and current events",

    # llm=ChatGroq(temperature=0, model_name="google/gemma-2bt")
    # llm=ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    # llm=ChatGroq(temperature=0, model_name="t5-small", provider="EleutherAI")
)
fact_checker = Agent(
    role="Fact Checker",
    goal="Verify the claims made in the news article",
    backstory="You are a meticulous fact-checker with years of experience in journalism",

    # llm=ChatGroq(temperature=0, model_name="google/gemma-2b")
    # llm=ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    # llm=ChatGroq(temperature=0, model_name="t5-small", provider="EleutherAI")
)
analyst = Agent(
    role="News Analyst",
    goal="Analyze the news article for potential bias or false information",
    backstory="You are a seasoned News analyst with expertise in identifying misinformation",
    # llm=ChatGroq(temperature=0, model_name="google/gemma-2b")
    # llm=ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
    # llm=ChatGroq(temperature=0, model_name="t5-small", provider="EleutherAI")

)
# Task
def create_tasks(news_article):
    return [
        Task(
            description="Research the main claims and entities mentioned in the news article",
            agent=researcher,
            expected_output="A detailed report on the main claims and entities in the article, including background information and context."
        ),
        Task(
            description="Fact-check the main claims in the article using reliable sources",
            agent=fact_checker,
            expected_output="A list of verified and debunked claims from the article, with references to reliable sources."
        ),
        Task(
            description="Analyze the article for potential bias, false information, or misleading content related to the US 2024 election",
            agent=analyst,
            expected_output="An analysis report detailing any identified biases, false information, or misleading content, with explanations and potential impacts on the election."
        )
    ]


if st.button("Analyze News", key="analyze_news_button"):
    if news_input:
        st.write("Analyzing...")
        
        # Create tasks
        tasks = create_tasks(news_input)
        # Create and run the crew
        crew = Crew(
            agents=[researcher, fact_checker, analyst],
            tasks=tasks,
            verbose=True
        )
        result = crew.kickoff()
        
        # Display the result
        st.subheader("Analysis Result")
        st.write(result)
else:
        st.warning("Please enter a news article to analyze.")


# Add a footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: #FAFAFA;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
  
    """,
    unsafe_allow_html=True
)