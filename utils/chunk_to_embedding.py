# %%
import json
import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from openai import OpenAI

# 自定义通义向量模型嵌入器（添加分批处理）
class TongyiEmbeddings:
    def __init__(self, model_name="text-embedding-v1", batch_size=25):
        self.model_name = model_name
        self.batch_size = batch_size  # 每批次处理的文本数量
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """分批次处理文本嵌入，每批次最多处理 self.batch_size 条文本"""
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
                # 可选：添加重试逻辑或跳过当前批次
                raise
                
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """处理单个查询文本的嵌入"""
        return self.embed_documents([text])[0]

# 从JSON文件加载分块数据并创建向量库
def create_vector_db_from_json(json_path, persist_dir="./sjtu_manual_db"):
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"共加载 {len(chunks)} 个文档块")
    documents = []
    
    for chunk in chunks:#chunk为字典
        content = chunk.get("page_content", "")#get方法获取字典中的键值对，若不存在则返回空字符串
        metadata = chunk.get("metadata", {})
        documents.append(Document(page_content=content, metadata=metadata))#json中的每个块转换为Document对象
    
    # 创建嵌入器，设置每批次处理25条文本
    embeddings = TongyiEmbeddings(model_name="text-embedding-v1", batch_size=25)
    
    # 分批次创建向量库
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )

    vectordb.add_documents(documents)
    vectordb.persist()
    
    print(f"向量库已保存到: {persist_dir}，共{len(documents)}个文档")
    return vectordb

# 加载已保存的向量库用于检索
def load_vector_db(persist_dir="./sjtu_manual_db"):
    embeddings = TongyiEmbeddings(model_name="text-embedding-v1")
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    print(f"已加载向量库，包含 {vectordb._collection.count()} 个文档")
    return vectordb

# 示例：从JSON创建向量库
if __name__ == "__main__":
    json_path = r"..\data\langchain_structured_manual_4_chunks.json"
    vectordb = create_vector_db_from_json(json_path)

    # 执行检索示例
    query = "奖学金申请需要什么条件？"
    docs = vectordb.similarity_search(query, k=3)
    
    for i, doc in enumerate(docs):
        print(f"\n检索结果 {i+1}:")
        print(f"章节: {doc.metadata.get('chapter', 'N/A')}")
        print(f"内容: {doc.page_content[:200]}...")


