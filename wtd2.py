__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os


from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import streamlit as st
import os

st.title('🦜🔗 Querying')

if "file_name" not in st.session_state:
    st.session_state.file_name = ""


def loader():
    name = st.session_state.file_name
    file_path = f"test/{name}"
    #   directory = os.getcwd()  # Get the current working directory

    loader = UnstructuredExcelLoader(file_path, mode="paged")
    documents = loader.load()

    return documents


def split_and_embed(documents):

    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    # docs = text_splitter.create_documents(text)

    # Fix: was throwing error before
    # metadata only supports primitive types such as str, int, etc
    for doc in docs:
        doc.metadata["languages"] = "eng"

    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2")

    return embedding_function


def set_db(uploaded_file):
    documents = loader()
    # embeddings = split_and_embed(uploaded_file)
    embeddings = split_and_embed(documents)

    db = Chroma().from_documents(documents=documents, embedding=embeddings,
                                 persist_directory="/tmp/brcl_03", collection_metadata={"hnsw:space": "cosine"})
    places = set([doc.metadata["page_name"] for doc in documents])
    # places = set({'LIQB', 'OV1', 'CR8', 'CONTENTS', 'Catalog', 'CCR7', 'MR2-B', 'LIQ1', 'KM1'})
    places.remove("Catalog")

    prompt = PromptTemplate(template="""
    ---
    Context: 
    {context}
    ---


    Answer in following format:
    [
    extracted_words: [<extract the words {question} form the above context section if present else keep empty. Use fuzzy matching to extract>],
    answer: <yes or no by checking if fuzzy match of either of {question} is present in context, assuming that we do not want related words, just same words>,
    reasoning: <reason you think extracted_words are {question} are exact fuzzy match in 1 line, assuming that we do not want related words, just same words>,
    percentage_match: <x% based of fuzzy match between extracted_words and {question}>
    ]
    """, input_variables=["context"])

    for page_name in places:
        # print("\n----------------------------------------------------\n")
        # print(f"\nPage Name- {page_name}\n")
        st.write(f"\nPage Name- {page_name}\n")
    # print("\n----------------------------------------------------\n")

        attributes = ["CCR or Counterparty credit risk", "F-IRB or Foundation internal ratings-based Approach", "IRC or Incremental Risk Charge", "A-IRB or Advance internal ratings-based Approach",
                      "SREP or Supervisory Review and Evaluation Process", "TURF or Total Unduplicated Reach and Frequency", "LCR or Liquidity Coverage Ratio", "HLBA or historical look-back approach", "RWEA"]

        for query in attributes[:2]:
            retriever = db.as_retriever(search_type="mmr",
                                        search_kwargs={'filter': {
                                            'page_name': page_name}, 'k': 3}
                                        )
            llm = OpenAI(temperature=0)  # model_name="text-davinci-003"
            chain = RetrievalQA.from_chain_type(llm=llm,
                                                chain_type="stuff",
                                                retriever=retriever,
                                                chain_type_kwargs={"prompt": prompt})

            response = chain({'query': query})
            # print(f"{response['query']}")
            # print(f"{response['result']}")
            st.write(f"{response['query']}")
            st.write(f"{response['result']}")
            # print("\n\n\n\n")


def save_uploaded_file(uploadedfile):
    file_name = uploadedfile.name
    st.session_state.file_name = file_name
    with open(os.path.join("test", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved file :{} in tempDir".format(uploadedfile.name))


def file_upload_form():
    with st.form('fileform'):
        supported_file_types = ["xlsx"]
        uploaded_file = st.file_uploader(
            "Upload a file", type=(supported_file_types))
        st.write(uploaded_file)
        submitted = st.form_submit_button("Submit")
        # st.write(uploaded_file.path)
        if submitted:
            if uploaded_file is not None:
                if uploaded_file.name.split(".")[-1] in supported_file_types:
                    # set_LLM(uploaded_file)
                    # st.session_state.current_filename = uploaded_file.name
                    st.write("File Uploaded successfully")

                    file_details = {"FileName": uploaded_file.name,
                                    "FileType": uploaded_file.type}

                    save_directory = "test"
                    os.makedirs(save_directory, exist_ok=True)
                    st.write("Directory created successfully")

                    save_uploaded_file(uploaded_file)
                    
                    # Process and save the uploaded file to the desired location
                    # process_and_save_file(uploaded_file, save_directory)

                    set_db(uploaded_file)
                else:
                    st.write(
                        f"Supported file types are {', '.join(supported_file_types)}")
            else:
                st.write("Please select a file to upload first!")
                # with st.spinner('Generating...'):
                #     generate_response(query_text, filename)


file_upload_form()
