# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json
from pathlib import Path

# %%


def chunk_structured_json(json_path, chunk_size=300, chunk_overlap=50):
    """
    将结构化JSON文件拆分为带元数据的文本块(chunk)
    
    Args:
        json_path: 结构化JSON文件路径
        chunk_size: 每个chunk的最大字符数
        chunk_overlap: 相邻chunk的重叠字符数
    """
    # 1. 读取结构化JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        structured_data = json.load(f)
    
    # 2. 初始化文本分割器（按语义分割）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ";", ",", " ", ""]  # 优先按自然分隔符拆分
    )
    
    # 3. 遍历章节和条款，生成chunk
    documents = []
    for chapter in structured_data["content"]:
        chapter_title = chapter["chapter"]  # 章节标题（如"第一章 总则"）
        
        for clause in chapter["clauses"]:
            # 条款元数据（用于追溯来源）
            metadata = {
                "document_title": structured_data["metadata"]["document_title"],
                "source": structured_data["metadata"]["source"],
                "chapter": chapter_title,
                "clause_id": clause["clause_id"],  # 条款ID（如"第三十九条"）
                "clause_title": clause["title"]    # 条款标题（如"医学院校区变更"）
            }
            
            # 构建条款完整文本（合并标题、内容、子项）
            clause_parts = []
            # 添加条款ID和标题
            if clause["title"]:
                clause_parts.append(f"{clause['clause_id']} {clause['title']}")
            else:
                clause_parts.append(f"{clause['clause_id']}")
            # 添加条款内容
            clause_parts.append(clause["text"])
            # 添加子项（如果有）
            if clause["sub_items"]:
                clause_parts.extend([f"({i+1}){sub_item}" for i, sub_item in enumerate(clause["sub_items"])])
            
            # 合并为完整文本
            full_text = "\n".join(clause_parts)
            
            # 4. 拆分当前条款为chunk（如果文本过长）
            # 将文本转换为LangChain的Document对象（便于分割器处理）
            doc = Document(page_content=full_text, metadata=metadata)
            # 分割为chunk
            chunks = text_splitter.split_documents([doc])
            
            # 添加到文档列表
            documents.extend(chunks)
    
    # 5. 保存chunk结果（可选，便于查看）
    output_path = Path(json_path).parent / f"{Path(json_path).stem}_chunks.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        # 转换为字典格式（Document对象无法直接JSON序列化）
        serialized = [{
            "page_content": doc.page_content,
            "metadata": doc.metadata
        } for doc in documents]
        json.dump(serialized, f, ensure_ascii=False, indent=2)
    
    print(f"已生成 {len(documents)} 个chunk,保存至:{output_path}")
    return documents

# 使用示例：处理之前生成的结构化JSON
if __name__ == "__main__":
    # 输入：之前生成的结构化JSON路径
    structured_json_path = Path("langchain_structured_manual_4.json")
    # 生成chunk
    chunks = chunk_structured_json(
        json_path=structured_json_path,
        chunk_size=300,  # 适合规章制度的短条款
        chunk_overlap=50
    )


