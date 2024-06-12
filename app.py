import os
import streamlit as st 
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.agents import initialize_agent, load_tools, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

load_dotenv()

st.title("langchain-streamlit-app")

# st.session_stateにmessagesがない場合、空のリストで初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# st.session_state.messagesでループし、保存されているテキストを表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザーからの入力を受け取る
prompt = st.chat_input("What is up?")
print(prompt)

# エージェントチェーンの作成関数
def create_agent_chain():
    chat = ChatOpenAI(
        model_name=os.environ["OPENAI_API_MODEL"],
        temperature=float(os.environ["OPENAI_API_TEMPERATURE"]),
        streaming=True,
    )

    # OpenAI Functions AgentのプロンプトにMemoryの会話履歴を追加するための設定
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }

    # OpenAI Functions Agentが使える設定でMemoryを初期化
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    tools = load_tools(["ddg-search", "wikipedia"])
    return initialize_agent(
        tools, chat, agent=AgentType.OPENAI_FUNCTIONS, agent_kwargs=agent_kwargs, memory=memory
    )

# エージェントチェーンの初期化を一度だけ行う
if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = create_agent_chain()

# 入力された文字列がある場合、ユーザーの入力内容をst.session_state.messagesに追加
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):  # ユーザーのアイコンで
        st.markdown(prompt)  # promptをマークダウンとして整形して表示

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        response = st.session_state.agent_chain.run(prompt, callbacks=[callback])
        
        if response.strip() != "complete!":
            st.markdown(response)

        # 応答をst.session_state.messagesに追加
        st.session_state.messages.append({"role": "assistant", "content": response})
