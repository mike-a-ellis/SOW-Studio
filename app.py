import streamlit as st
st.set_page_config(page_title="SOW Studio", page_icon="✒️")
st.title("✒️ SOW Studio: LLM-Powered Statements of Work Authoring Tool")

st.markdown(f"""

            ## Overview
SOW Studio is a chat based solution designed to augment the creation of Statements of Work (SOW) for IT consulting firms. By integrating Large Language Models (LLMs) with a comprehensive database of exemplary SOWs, SOW Studio offers increased accuracy, efficiency, and consistency in SOW authoring. This tool not only streamlines the writing process but also ensures adherence to best practices and industry standards, significantly reducing errors and improving scalability for businesses.

#### 1. **SOW Filing Agent**
The `sow_filing_agent` serves as a robust query system for existing SOWs, allowing users to access and replicate sections from well-crafted examples. It supports queries across various SOW components such as service descriptions, team investments, project schedules, constraints, and more. This agent is invaluable for users seeking to enhance the quality of their SOWs with proven structures and terms, offering guidance on everything from application development projects to management training sessions.

#### 2. **Sentiment Analysis Agent**
The `sentiment_analysis_agent` provides sophisticated sentiment analysis capabilities, enabling users to assess and adjust the emotional tone of SOW texts. Whether you need to make your SOW sound more positive, neutral, or negative, this agent assists in fine-tuning the language to better align with the intended audience and company ethos.

#### 3. **SOW Grammar Analysis Agent**
Our `sow_grammar_analysis_agent` enhances the clarity and professionalism of SOW documents by checking grammar, dates, and verbiage. This agent is essential for final proofreading and error correction, ensuring that each SOW is not only structurally sound but also impeccably presented. From grammatical tweaks to date verifications, this agent guarantees that your SOWs are polished and ready for presentation.

### Benefits and Impact
By leveraging SOW Studio, IT consulting firms can achieve a higher standard of document precision and effectiveness, leading to clearer agreements,  smoother project initiations, and decreased labor costs during the authoring process itself.. The integration of LLM technology with practical tools for document creation and analysis provides a competitive edge in the consulting industry, ultimately facilitating better business outcomes and client relationships.

""")

from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.text_splitter import RecursiveCharacterTextSplitter

from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader

import os
from dotenv import load_dotenv,find_dotenv
import json
import tiktoken

load_dotenv(find_dotenv())

openai_chat_model_3_5 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,openai_api_key=os.getenv("OPENAI_API_KEY"))
openai_chat_model_4 = ChatOpenAI(model="gpt-4-turbo", temperature=0,openai_api_key=os.getenv("OPENAI_API_KEY"))

primary_qa_llm = openai_chat_model_4

def tiktoken_len(text):

    tokens = tiktoken.encoding_for_model(primary_qa_llm.model_name).encode(
        text,
    )
    return len(tokens)

@st.cache_data
def split_chunk_text(input_path="./input_data/"):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function = tiktoken_len,
    )
    
    loader = DirectoryLoader(input_path, glob="**/*.txt")
    docs = loader.load_and_split(text_splitter=text_splitter)
    return docs


def call_sow_filing_agent(text):
    texts = split_chunk_text()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = FAISS.from_documents(texts, embeddings)
    retriever = vector_store.as_retriever(search_type="mmr",search_kwargs={'k': 6})

    template = """

    Background and Context:
    A statement of work (SOW) is a formal document that captures and defines the work activities, deliverables, and timeline a vendor or service provider must execute in performance of specified work for a client. SOWs typically include sections like:

    Introduction and scope
    Description of required services/deliverables
    Timelines and schedule
    Standards and requirements
    Pricing and invoicing
    Other terms and conditions

    The purpose is to clearly articulate all components of the work agreement between a client organization and a vendor. SOWs are commonly used in outsourcing, consulting, professional services, and other contracted project work.
    Prompt:
    You are an expert consultant on developing effective statements of work for professional services engagements. Using the given context about SOWs provided above, along with any other relevant knowledge you may have, please assist with the following types of requests:

    Explain the typical structure and sections included in a SOW.
    Provide examples of well-written SOW language for specific sections like scope, deliverables, schedule, requirements, etc.
    Analyze excerpts or full samples of SOWs and provide feedback on areas that are clear vs. ambiguous, complete vs. lacking details, etc.
    Answer general questions about SOW best practices, common pitfalls to avoid, tips for well-defined and measurable deliverables, etc.

    For any request, draw upon the context provided as well as your own knowledge about SOWs and contracting for professional services.

    Answer the question based only on the following context if it is a question.  If it is not a question, but asking for examples using the provided context to provide examples.  All output should be detailed.  If you do not have enough information to answer accurately, please respond with 'I don't know'. 

        Context:
        {context}

        Question:
        {question}
    
    """

    prompt = ChatPromptTemplate.from_template(template)
    
    retrieval_augmented_qa_chain = (
        # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
        # "question" : populated by getting the value of the "question" key
        # "context" : populated by getting the value of the "context" key and chaining it into the base_retriever
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        # "context*" : is assigned to the "context" retrieved from the previous step
        #               by getting the value of the "context" key from the previous step
        | RunnablePassthrough.assign(context=itemgetter("context"))
        # "response" : the "context" and "question" values are used to format our prompt object and then piped
        #              into the LLM and stored in a key called "response"
        | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
        # "context" populated by getting the value of the "context" key from the previous step
        )

    result = retrieval_augmented_qa_chain.invoke({"question" : text})
    response_str = result['response'].content

    with st.expander("Retrieved Context"):
        st.write(result["context"])

    return response_str

def call_sentiment_analysis_agent(text):

    prompt = """Perform sentiment analysis on the given text to classify it as positive, negative, or neutral within a business context. Utilize your knowledge of natural language processing to identify sentiment-bearing words and phrases. Provide a detailed, step-by-step explanation for your sentiment assessment, clearly stating the sentiment category and reasoning.
                If it is negative provide an example of how it could be rewritten to make it more positive.

                Examples:

                Text: "The new product launch was a disappointment, and sales figures were below expectations."
                Sentiment: Negative
                Explanation: The words "disappointment" and "below expectations" carry negative connotations, indicating an unfavorable outcome or dissatisfaction with the product launch and sales performance.

                Text: "Our customer support team consistently receives excellent feedback for their prompt and friendly service."
                Sentiment: Positive
                Explanation: The phrases "excellent feedback" and "prompt and friendly service" convey positive sentiment, suggesting satisfaction and praise for the customer support team's performance.

                Text: "The quarterly financial report showed steady growth and met projected targets."
                Sentiment: Neutral
                Explanation: While the phrases "steady growth" and "met projected targets" are generally positive, the overall tone is factual and objective, lacking strong positive or negative language, resulting in a neutral sentiment.

                For the given text:
                ----
        {question}
    
    """

    prompt = ChatPromptTemplate.from_template(prompt)

    chain = prompt | primary_qa_llm

    result = chain.invoke({"question" : text})
    #print(result['content'])
    response_str = result.content
    return response_str

def call_grammar_analysis_agent(text):

    prompt = """You are a professional SOW analyst and proofreader and copywriter. Your role is to review text from statements of work and provide corrections and improvements in the following areas:

                Grammar and spelling and word choice:


                Fix any grammatical errors, typos, or misspellings in the text.
                Check for proper punctuation usage.
                Numbers less than 10 should be written as words, followed by the numeral in parenthesis.  For example '9' should be written as 'nine (9)'.
                Identify areas where the wording is unclear, ambiguous, or could be improved.
                Suggest alternative phrasing to improve clarity and specificity.
                Watch for instances of jargon, legalese, or unnecessary complexity and simplify where possible.


                Date and timeline review:


                Verify that all dates, deadlines, and timeline references in the SOW are accurate and consistent.
                Flag any conflicting or ambiguous date/time information.
                Flag any dates that are not within 60 days of today's date.

                Consistency:


                Ensure consistent use of terms, phrases, formatting, etc. throughout the document.
                Check that defined terms are used properly.

                When reviewing SOW text, provide your analysis and suggestions in the following format:

                Corrections:

                [Grammar/spelling/word choice corrections]
                [Date/timeline fixes]
                [Consistency changes]

                Explanation:
                [Brief explanation of the types of changes made and why, for each category]

                Here is the chat input containing the sow text to be analyzed.
                ---
                {question}
    
    """

    prompt = ChatPromptTemplate.from_template(prompt)

    chain = prompt | primary_qa_llm

    result = chain.invoke({"question" : text})
    response_str = result.content
    return response_str

def init_router_chain():
    router_system_prompt = f"""
    You are a world class analyst agent orchestrator. Your role is to assist the users by routing the questions to the right agent.

    Given the user question below and histort of the conversation, classify the intent to be served by one of the following agents:

    ##1
    "agent_name": "sow_filing_agent",
    "agent_description": "This agent can query existing Statements of Work or SOWs.  It can be used to provide examples of known good SOW or SOW headings such as Description of Services and Deliverables, Team and Investment, 
                         Project Schedule for Delivery and Payment, Project Constraints and Flexibility, Expiration, Travel Expenses, and Assumptions. For example, Give me an example Description of Services and Deliverables.  Do we have any SOWs regarding SQL Server or Project Management or Training?

    ##2
    "agent_name": "sentiment_analysis_agent"
    "agent_description": "This agent can provide a sentiment analysis of text and make provided text more positive, neutral, or negative in sentiment.  For example, Is this following text positive or negative in emotion or sentiment?  Rewrite the provided text to be more positive. Rewrite the provided text to be more neutral.  What words the following text positive or negative or neutral in sentiment?"

     ##3
    "agent_name": "sow_grammar_analysis_agent"
    "agent_description": "This agent can provide a an analysis of an SOW to proof read, check dates, make grammatical corrections, and fix verbiage.  It is used when some example statement of work SOW text is uploaded for correction or checking.  For example, Correct the grammar for the following SOW.  Check dates on this SOW. Find and fix word flaws or optimizations in the included text."

    #### USER QUESTION

    ### INSTRUCTIONS:
    Some questions may be in reference to previous questions in the history.

    #### RESPONSE FORMAT
    Only respond in the JSON format with the following keys:

    "agent_name":"<classified agent name>",

    ALWAYS use one key, though the value can be empty.
    RESPONSE SHOULD ALWAYS BE IN JSON FORMAT.

    """

    router_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", router_system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ]
    )

    chain = (
        router_prompt | ChatOpenAI() | StrOutputParser()
    )

    chain_with_history = RunnableWithMessageHistory(chain, lambda session_id: msgs, input_messages_key="question", history_messages_key="history")

    return chain_with_history

router_chain = init_router_chain()

def call_router_chain(router_chain, question):
    response_str = router_chain.invoke({"question":question}, config={"configurable": {"session_id": "any"}})
    return response_str

msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("human").write(prompt)

    response = call_router_chain(router_chain, prompt)
    response = json.loads(response)

    agent_name = response["agent_name"]

    print(f"Agent Name: {agent_name}")

    st.chat_message("ai").write(f"Calling Agent - {agent_name} with your prompt:")
    
    if(agent_name == "sow_filing_agent"):
        agent_response = call_sow_filing_agent(prompt)
        st.chat_message("ai").write(agent_response)
    elif(agent_name == "sentiment_analysis_agent"):
        agent_response = call_sentiment_analysis_agent(prompt)
        st.chat_message("ai").write(agent_response)
    elif(agent_name == "sow_grammar_analysis_agent"):
        agent_response = call_grammar_analysis_agent(prompt)
        st.chat_message("ai").write(agent_response)
    else:
        st.chat_message("ai").write("No tool was chosen.  Error.  Retry with a different prompt.")
        
with view_messages:
    view_messages.json(st.session_state.langchain_messages)
    