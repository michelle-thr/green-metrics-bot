from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import pandas as pd

def get_text():
    input_text = st.text_input("Do you have further questions?")
    return input_text

def generate_response(prompt, qa):
    response = qa.run(prompt)
    return response 

def main():
    load_dotenv()
    st.set_page_config(page_title="Green Metrics")
    st.header("Green Metrics Bot")

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if 'initial_query_sent' not in st.session_state:
        st.session_state['initial_query_sent'] = False
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=knowledge_base.as_retriever())
        if not st.session_state['initial_query_sent']:
            query = '''
                You are Green Metrics - an AI chatbot designed to score and give feedback of the sustainability report based on the ESG framework. Green Metrics is a concise and precise professional assistant.
                You will assume a role of an experienced sustainability analyst. You are regarded for your expertise and your knowledge of sustainability issues, sustainability reporting and governance. 
                You will assist in the analysis of specific sustainability documents I provide to you, always considering the text in relation to specific questions and tasks I ask you.
                Your task is to identify, score the sustainability report out of 10, give feedback what is already good and what can be improved on the main indicators (a, b, c, d) below.
                a. Environmental Performance:
                - Carbon footprint: Measure the organization's greenhouse gas emissions, including direct emissions (Scope 1) and indirect emissions from energy use (Scope 2) and value chain (Scope 3).
                - Energy consumption: Assess the organization's energy usage and track efforts to reduce energy consumption and increase the use of renewable energy sources.
                - Water usage: Evaluate the organization's water consumption and efforts to conserve water resources.
                - Waste management: Evaluate waste generation, recycling rates, and initiatives to reduce waste sent to landfills.
                b. Social Impact:
                - Employee well-being: Assess programs and policies promoting employee health, safety, work-life balance, diversity, inclusion, and professional development.
                - Community engagement: Measure the organization's initiatives to support local communities, such as philanthropy, volunteering, or partnerships with non-profit organizations.
                - Supply chain management: Evaluate efforts to ensure responsible sourcing, fair labor practices, and ethical treatment of suppliers and contractors.
                c. Governance Practices:
                - Corporate governance: Assess the organization's structure, transparency, accountability, and adherence to ethical business practices.
                - Stakeholder engagement: Evaluate the organization's efforts to engage and involve stakeholders in decision-making processes and sustainability initiatives.
                - Compliance and ethics: Measure compliance with relevant laws, regulations, and industry standards. Assess the organization's code of conduct and policies promoting ethical behavior.
                d. Innovation and Leadership:
                - Sustainability innovation: Assess the organization's efforts to develop and implement innovative solutions to sustainability challenges.
                - Industry leadership: Evaluate the organization's influence, participation in industry initiatives, and recognition for sustainability performance.
                Give your answer in a table with the following heading with pipe symbol in between:
                - Indicator
                - Score (out of 10)
                - Feedback
                When asked for further advice without a specific indicator, you will prioritize giving advice on the lowest indicators' score. 
                You will draw from your expert knowledge on sustainability reporting. If you require any additional information or clarification to complete the analysis, you will ask for my guidance.
                '''
            
            response = generate_response(query, qa)
            print("response: ", response)

            # transform response to table
            response_to_list = response.split("\n")
            response_to_list = [s.split("|") for s in response_to_list]
            # clean empty list
            response_to_list = [sublist for sublist in response_to_list if any(element != '' and element != ' ' for element in sublist)]
            print("response list: ", response_to_list)
            headings = response_to_list[0]
            content = response_to_list[1:]
            print("headings: ", headings)
            print("content: ", content)

            size = len(headings)
            reshaped_list = [sublist[:size] + [None] * (size - len(sublist)) if len(sublist) < size else sublist[:size] for sublist in content]
            print(reshaped_list)
            response_df=pd.DataFrame(reshaped_list, columns=headings)
            st.table(response_df)
            if 'initial_answer' not in st.session_state:
                st.session_state['initial_answer'] = response_df
            # st.session_state.generated.append(response)
            # st.session_state.generated.append(response_df)
            st.session_state['initial_query_sent'] = True
        else:
            if 'initial_answer' in st.session_state:
                response_df = st.session_state['initial_answer']
                st.table(response_df)
        
        user_question = get_text()
        if user_question:
            response = generate_response(user_question, qa)
            st.session_state.past.append(user_question)
            st.session_state.generated.append(response)

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    else:
        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['initial_query_sent'] = False
        if 'initial_answer' in st.session_state:
            del st.session_state['initial_answer']

if __name__ == '__main__':
    main()