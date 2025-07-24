# %%
import re
import json
from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_structured_data(structured_data):
    """清洗结构化JSON数据，优化后续chunk拆分效果"""
    # 清洗元数据
    metadata = structured_data.get("metadata", {})
    if "document_title" in metadata:
        metadata["document_title"] = re.sub(r"[\s]+", " ", metadata["document_title"]).strip()
    if "source" in metadata:
        metadata["source"] = metadata["source"].strip()
    structured_data["metadata"] = metadata
    
    # 遍历章节清洗
    cleaned_chapters = []
    for chapter in structured_data.get("content", []):
        # 清洗章节标题
        chapter_title = chapter.get("chapter", "")
        chapter_title = re.sub(r"^第[一二三四五六七八九十]+章\s*", lambda m: m.group(0).strip() + " ", chapter_title)
        chapter_title = re.sub(r"\s+", " ", chapter_title).strip()
        chapter["chapter"] = chapter_title
        
        # 遍历条款清洗
        cleaned_clauses = []
        for clause in chapter.get("clauses", []):
            # 清洗条款ID
            clause_id = clause.get("clause_id", "")
            clause_id = re.sub(r"[^\u4e00-\u9fa50-9]", "", clause_id).strip()
            clause["clause_id"] = clause_id
            
            # 清洗条款标题
            clause_title = clause.get("title", "")
            clause_title = re.sub(r"\s+", " ", clause_title).strip()
            clause_title = re.sub(r"^[:：\s]+", "", clause_title)
            clause["title"] = clause_title
            
            # 清洗条款主文本
            clause_text = clause.get("text", "")
            clause_text = re.sub(r"\s+", " ", clause_text).strip()
            clause_text = re.sub(r"[#*_]+", "", clause_text)
            clause["text"] = clause_text
            
            # 清洗子项
            sub_items = clause.get("sub_items", [])
            cleaned_sub_items = []
            # 仅当sub_items是列表时才遍历
            if isinstance(sub_items, list):
                for sub in sub_items:
                    if not sub:
                        continue
                    sub_clean = re.sub(r"^（?[一二三四五六七八九十]+）?\s*", "", sub)
                    sub_clean = re.sub(r"\s+", " ", sub_clean).strip()
                    cleaned_sub_items.append(sub_clean)
            # 保持None值
            clause["sub_items"] = cleaned_sub_items if cleaned_sub_items else (sub_items if sub_items is None else [])
            
            cleaned_clauses.append(clause)
        
        chapter["clauses"] = cleaned_clauses
        cleaned_chapters.append(chapter)
    
    structured_data["content"] = cleaned_chapters
    return structured_data

# 整合清洗+chunk拆分的完整流程
def process_and_chunk(json_path, chunk_size=300, chunk_overlap=50):
    """先清洗结构化JSON，再拆分chunk"""
    # 1. 读取原始结构化JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        structured_data = json.load(f)
    
    # 2. 清洗数据
    cleaned_data = clean_structured_data(structured_data)
    
    # 3. 保存清洗后的JSON
    cleaned_json_path = Path(json_path).parent / f"{Path(json_path).stem}_cleaned.json"
    with open(cleaned_json_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    print(f"清洗完成，保存至：{cleaned_json_path}")
    
    # 4. 拆分chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ";", ",", " ", ""]
    )
    
    documents = []
    for chapter in cleaned_data["content"]:
        chapter_title = chapter["chapter"]
        for clause in chapter["clauses"]:
            metadata = {
                "document_title": cleaned_data["metadata"]["document_title"],
                "source": cleaned_data["metadata"]["source"],
                "chapter": chapter_title,
                "clause_id": clause["clause_id"],
                "clause_title": clause["title"]
            }
            
            # 拼接清洗后的条款内容
            clause_parts = []
            if clause["title"]:
                clause_parts.append(f"{clause['clause_id']} {clause['title']}")
            else:
                clause_parts.append(f"{clause['clause_id']}")
            clause_parts.append(clause["text"])
            if clause["sub_items"]:
                clause_parts.extend([f"（{i+1}）{sub_item}" for i, sub_item in enumerate(clause["sub_items"])])
            
            full_text = "\n".join(clause_parts)
            doc = Document(page_content=full_text, metadata=metadata)
            chunks = text_splitter.split_documents([doc])
            documents.extend(chunks)
    
    # 保存chunk结果
    chunk_output_path = Path(json_path).parent / f"{Path(json_path).stem}_chunks.json"
    with open(chunk_output_path, 'w', encoding='utf-8') as f:
        serialized = [{
            "page_content": doc.page_content,
            "metadata": doc.metadata
        } for doc in documents]
        json.dump(serialized, f, ensure_ascii=False, indent=2)
    
    print(f"已生成 {len(documents)} 个chunk，保存至：{chunk_output_path}")
    return documents


# 使用示例
if __name__ == "__main__":
    structured_json_path = Path(r"E:\py_vscode_project\RAG_rule_chatrobot\varied_formats\md_langchain_structured_manual_4.json")  # 原始结构化JSON路径
    chunks = process_and_chunk(
        json_path=structured_json_path,
        chunk_size=300,
        chunk_overlap=50
    )


