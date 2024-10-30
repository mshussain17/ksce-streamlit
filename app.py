import os
import streamlit as st
from io import StringIO
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.retrievers import SVMRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv 
import streamlit as st
# from st_files_connection import FilesConnection
import boto3
import tempfile
import shutil
import re
import time

load_dotenv()  


# Streamlit page configuration
st.set_page_config(page_title="KSCE-GPT", page_icon="📄")
st.title("KSCE-GPT")

import common as common
from langchain_openai import OpenAIEmbeddings
import os
from rerank import compression_retriever
import anthropic
import logging
from copy import deepcopy
openai_embeddings = OpenAIEmbeddings(model=common.EMBEDDING_MODEL)

logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s', level=logging.INFO)
logger = logging.getLogger('api.py')

# Anthropic API Key:
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Client(api_key=anthropic_key)

# Retrieval tool:
def retrieval(query):
    documents = compression_retriever.invoke(query)
    doc = deepcopy(documents)
    print("doc metadata")
    print([d.metadata['content_id'] for d in doc])
    # for document in doc:
    #     document.metadata.pop('images')
    # print(doc[0])
    relevant_doc = {"page_content": documents[0].page_content,"metadata":documents[0].metadata}

    model_name = common.TOOL_MODEL
    message = client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0.0,
        system=f"""You are a civil engineering expert. Here are the documents in the form of a python list:
                       {doc}
                       In each document there is a key content_id in the metadata dictionary. For the given query provide the content_id from the metadata dictionary of the document that answers the query. Make sure to match important terms from the civil engineering industry. Output just the content_id. If no document is relevant simply output 0 and nothign else. Only output one content_id or 0. Do not include any explanation.""",
        messages=[
            {"role": "user", "content": query}
        ]
    )
    try:
        print("In first try block. id:")
        id = message.content[0].text.strip()
        print(id)
    except:
        try:
            message = client.messages.create(
                model=model_name,
                max_tokens=1024,
                temperature=0.0,
                system=f"""You are a civil engineering expert. Here are the documents in the form of a python list:
                       {doc}
                       In each document there is a key content_id in the metadata dictionary. For the given query provide the content_id from the metadata dictionary of the document that answers the query. Make sure to match important terms from the civil engineering industry. Output just the content_id. If no document is relevant simply output 0 and nothign else. Only output one content_id or 0. Do not include any explanation.""",
                messages=[
                    {"role": "user", "content": query}
                ]
            )
            print("In second try block. id:")
            id = message.content[0].text.strip()
            print(id)
        except:
            print("In last except block")
    # print("Final ID = " + id)
    return id, documents

def ask_ai(question: str):
    logger.info(f"Received question: {question}")
    messages=[]
    messages.append({"role": "user", "content": question})
    model_name = common.TOOL_MODEL
    tool = common.SEARCHING_TOOL
    initial_response = client.beta.tools.messages.create(
        model=model_name,
        max_tokens=1024,
        tools=tool,
        system="""You are a helpful Korean assistant that answers questions regarding civil engineering industry issues using the context provided to you. You speak only Korean language and converse when necessary, but politely refuse questions that are out of scope. If the question asked is irrelevant or out of scope output a short message saying 'Sorry I can't answer the question.'. Use retrieval tool to retrieve documents relevant to construction. Always use the tool to answer questions regarding civil engineering industry.""",
        messages=messages
    )

    def process_tool_call(tool_name, tool_input):
        if tool_name == "retrieval":
            resp = retrieval(tool_input["query"])
            return resp

    if initial_response.stop_reason == "tool_use":
        tool_use = next(block for block in initial_response.content if block.type == "tool_use")
        tool_name = tool_use.name

        tool_input = tool_use.input
        tool_result, source_documents = process_tool_call(tool_name, tool_input)
        id = ""

        if tool_result == "0":
            return "No source found matching the query", [], "", False
        else:
            id = tool_result
            tool_result = [doc.page_content for doc in source_documents if doc.metadata['content_id'] == id][0]

        messages.append(
            {
                "role": "assistant",
                "content": initial_response.content
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": str(tool_result)
                    }
                ],
            }
        )
        response = client.beta.tools.messages.create(
        model=model_name,
        max_tokens=2000,
        temperature=0,
        system="""You are a helpful Korean assistant that answers questions regarding civil engineering issues using the context provided
                    to you. You speak only Korean language and converse when necessary, but politely refuse questions that are out of scope. Use retrieval tool to retrieve relevant docuemnts. Always use the tool to answer questions regarding construction industry.
                    Keep your answer concise and to the point. Use context as your source of response and exclude any extra information from the context that doesn't support query. Do not add any information on your own.
                    """,
        messages=messages,
        tools=tool
    )
        output_message = response.content[0].text.replace('<result>\n', '').replace('\n</result>', '')
        print(output_message)
        retrieved_document = [doc for doc in source_documents if doc.metadata['content_id'] == id][0]
        images = retrieved_document.metadata['images']
       
        return output_message, images, retrieved_document.metadata['file_name'], True
        
    else:
        
        response = initial_response
        final_response = response.content[0].text.replace('<result>\n', '').replace('\n</result>', '')
        messages.append({"role": "assistant", "content": response.content[0].text})
        print(response.content[0].text.replace('<result>\n', '').replace('\n</result>', ''))

        return final_response, [], "", False


@st.cache_resource
def create_retriever(_embeddings, splits, retriever_type):
    if retriever_type == "SIMILARITY SEARCH":
        vectorstore = FAISS.from_texts(splits, _embeddings)
        retriever = vectorstore.as_retriever(k=5)
    elif retriever_type == "SUPPORT VECTOR MACHINES":
        retriever = SVMRetriever.from_texts(splits, _embeddings)
    return retriever

@st.cache_resource
def split_texts(text, chunk_size, overlap):
    st.info("Splitting document...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()
    return splits
    
from langchain_community.vectorstores import FAISS
import tempfile
from botocore.exceptions import ClientError, NoCredentialsError
import os
import boto3
from aws_functions import download_files_from_s3, check_file_exists_in_s3
from PIL import Image

if user_query := st.chat_input("Ask your question") : 
    # import pdb; pdb.set_trace()
    if user_query :
        st.write(f"QUESTION : {user_query}")
    # Retrieve relevant information
    # if 'response' not in st.session_state : 
        # st.session_state.response = ""
    # st.session_state.response = get_groq_response(user_query, context, selected_model)

    placeholder = st.empty()
    response, images, doc, tool_use = ask_ai(user_query)
    doc_url = "https://dgpt-public.s3.eu-north-1.amazonaws.com/ksce-public/" + doc
    images_urls = ["https://dgpt-public.s3.eu-north-1.amazonaws.com/ksce-public/" + x for x in images]
    st.write(response)
    updated_resp = ""
    if doc != "":
        with st.sidebar : 
            with st.container() :
                st.header("Generated answer references" , divider = "red")
                with st.expander(str(doc_url)) : 
                    st.write(doc_url , unsafe_allow_html = True)
    for image_path in images_urls:
        try:
            st.image(image_path, caption='', use_column_width=True)
        except:
            pass
    ###################################################################################  
    # time.sleep(1) 
    # placeholder.text_are("Response : " , "Here's a bouquet &mdash;\
                        #   :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")
    placeholder.write(updated_resp , unsafe_allow_html = True)