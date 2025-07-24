#from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI#58
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser#用于将大模型的输出解析为字符串
from langchain.memory import ConversationBufferMemory
from openai import OpenAI
import os
from typing import List#用于类型的注解
from langchain.schema.runnable import RunnableLambda
import base64  # 用于PDF文件编码
#复用自定义类
class TongyiEmbeddings:
    def __init__(self, model_name="text-embedding-v1", batch_size=25):
        self.model_name = model_name
        self.batch_size = batch_size
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            print(f"正在处理批次 {i//self.batch_size + 1}/{total_batches}，包含 {len(batch_texts)} 条文本")

            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch_texts,
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                print(f"批次 {i//self.batch_size + 1} 处理失败: {e}")
                raise

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

@st.cache_resource#防止重复加载向量库
def load_vector_db():
    embeddings = TongyiEmbeddings(model_name="text-embedding-v1")
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=r"..\utils\sjtu_manual_db" 
    )
    return vectordb

def init_qwen_model():#返回ChatOpenAI模型实例
    return ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-max",
        temperature=0.3#数字越小，回答越固定
    )

prompt_template = """
你是上海交通大学学生手册的问答助手，请根据以下参考内容，准确回答用户的问题。
如果参考内容中没有相关信息，请回复"抱歉，未找到相关规定"。

参考内容：
{context}

对话历史：
{history}

用户问题：{question}

回答：
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

def create_chain(model):
    memory = ConversationBufferMemory(memory_key="history", input_key="question")
    
    def format_memory(inputs):
        memory_vars = memory.load_memory_variables({"input": inputs["question"]})
        return {
            "context": inputs["context"],
            "question": inputs["question"],
            "history": memory_vars["history"]
        }
    
    return RunnableLambda(format_memory) | prompt | model | StrOutputParser(), memory

def sync_messages_to_memory(memory, messages):
    memory.clear()
    for msg in messages:  #每个msg是一个字典，包含"role"和"content"
        if msg["role"] == "user":
            memory.chat_memory.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            memory.chat_memory.add_ai_message(msg["content"])

# 生成PDF文件下载链接的函数（本地文件）
def get_pdf_download_link(pdf_path):
    """将本地PDF文件编码为base64,生成可下载的链接"""
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    b64 = base64.b64encode(pdf_bytes).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="student_manual.pdf" target="_blank" style="display: block; text-align: center; padding: 8px; background-color: #007bff; color: white; border-radius: 4px; text-decoration: none;">下载学生手册PDF</a>'

# 生成图片按钮的HTML代码
def get_image_button(image_path, url, width=150):
    """生成包含图片的按钮链接"""
    # 读取图片并编码为base64
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode()
    
    # 返回HTML代码
    return f'''
    <a href="{url}" target="_blank" style="display: inline-block;">
        <img src="data:image/png;base64,{img_b64}" width="{width}" style="border: none; cursor: pointer;">
    </a>
    '''

def main():
    # 初始化对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 在代码中预设PDF地址（请修改为实际地址）
    PDF_URL = "https://jwc.sjtu.edu.cn/info/1476/12916.htm" #pdf来源地址

    # 设置图片按钮的路径和跳转链接
    IMAGE_PATH = r"..\img\官网.png"  # 图片文件路径
    SJTU_URL = "https://www.sjtu.edu.cn/"  # 上海交通大学官网

    pdf_file_path = r"..\data\SJTU2023版学生手册.pdf"   # 本地文件下载路径

    # 左侧侧边栏
    with st.sidebar:
        st.subheader("功能导航")
        
        try:
            st.markdown(get_pdf_download_link(pdf_file_path), unsafe_allow_html=True)
        except FileNotFoundError:
            st.error("本地PDF文件未找到,请检查路径是否正确")
        
        st.write("")  # 空行分隔
        
        try:
            st.markdown(f'<a href="{PDF_URL}" target="_blank" style="display: block; text-align: center; padding: 8px; background-color: #007bff; color: white; border-radius: 4px; text-decoration: none;">数据来源</a>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"URL格式错误: {e}")
        
        st.write("")  # 空行分隔

        # 图片按钮：访问上海交通大学官网
        if os.path.exists(IMAGE_PATH):
            st.markdown(get_image_button(IMAGE_PATH, SJTU_URL), unsafe_allow_html=True)
        else:
            st.error(f"图片文件未找到: {IMAGE_PATH}")

    # 主页面内容（保持不变）
    st.title("上海交大学生手册问答机器人")
    st.caption("基于学生手册内容的智能问答，支持学籍、奖学金等问题")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 重置对话"):
            st.session_state.messages = []
    with col2:
        with st.expander("📖 使用说明"):
            st.write("你可以对学生手册内容进行提问\n\n 提问示例：\n- 奖学金申请条件是什么？\n- 如何办理休学？\n- 考试作弊有什么处罚？")
    
    vectordb = load_vector_db()
    model = init_qwen_model()
    chain, memory = create_chain(model)
    sync_messages_to_memory(memory, st.session_state.messages)
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
    
    user_input = st.chat_input("请输入你的问题（例如：奖学金申请条件是什么？）")
    
    st.write("常见问题：")
    cols = st.columns(3)
    with cols[0]:
        if st.button("奖学金申请条件"):
            user_input = "奖学金申请条件是什么？"    
            st.session_state.messages.append({"role": "user", "content": user_input})
    with cols[1]:
        if st.button("休学办理流程"):
            user_input = "如何办理休学？"        
            st.session_state.messages.append({"role": "user", "content": user_input})
    with cols[2]:
        if st.button("考试作弊处罚"):
            user_input = "考试作弊有什么处罚？"
            st.session_state.messages.append({"role": "user", "content": user_input})

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        sync_messages_to_memory(memory, st.session_state.messages)

        with st.spinner("正在检索相关规定..."):
            docs = vectordb.similarity_search(user_input, k=4)#参考条目数量
            context = "\n\n".join([doc.page_content for doc in docs])#返回是一个字符串类型

        with st.spinner("正在生成回答..."):
            response = chain.invoke({
                "context": context,
                "question": user_input
            })
        
        with st.chat_message("assistant"):
            st.markdown(response)
            source_parts = ["参考文档:SJTU2023版学生手册"]
            if docs:
                chapters = []
                for doc in docs:
                    chapter = doc.metadata.get('chapter', '未知章节')
                    clause_id = doc.metadata.get('clause_id', '未知条款')
                    chapters.append(f"参考章节：{chapter}  {clause_id}")
                source_parts.extend(chapters)
            source_info = "\n\n".join(source_parts)
            st.caption(source_info)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        sync_messages_to_memory(memory, st.session_state.messages)

if __name__ == "__main__":
    main()