import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# 참조 데이터(PDF, Dox, PPT) 로딩을 위한 라이브러리
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# 몇개까지의 대화를 메모리에 넣어줄지 설정
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from streamlit_chat import message (메모리 구축을 위해 필요한 추가 라이브러리)
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config(
    page_title="DirChat",
    page_icon=":books:")

    st.title("_Private Data :red[QA Chat]_ :books:")        #_: 이텔릭체, red[] : 빨간색

    # converstaion, chat_history, processComplete 변수 정의
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # with 구문 : 구성 요소에 딸린 요소를 정하기 위한거
    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file", type=['pdf','docx', 'pptx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

    if process:     # 만약 Process 버튼이 눌리면
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
     
        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key)

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    # 메시지마다 with로 묶어줌 role에 따라서 markdown : 이미지(아이콘)과 메시지를 묶어서 추가
    # for : 메시지가 올라올때마다 작동
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 메모리에 저장하여 기억한뒤에 답변
    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):      # 사용자가 질문을 입력하면
        st.session_state.messages.append({"role": "user", "content": query})

        # 사용자 질문
        with st.chat_message("user"):
            st.markdown(query)

        # Assistant 답변
        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            
            with st.spinner("Thinking..."):     # 로딩시 progress를 표시하는 UI
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)    # help : ? 부분에 마우스를 올리면 text 띄움
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름(임시 저장 경로)으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")

        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),  # chat_history 저장, 답변에 해당하는 부분만 히스토리에 닮도록 설정
            get_chat_history=lambda h: h,   # 들어온 그대로 히스토리에 넣도록 설정
            return_source_documents=True,   # LLM이 참고한 문서를 출력하도록 설정
            verbose=True
        )
    return conversation_chain

if __name__ == '__main__':
    main()
