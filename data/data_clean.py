# %%
import re
import json
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def clean_text(text):
    """清理文本中的冗余信息"""
    #re.sub字符串中执行替换的操作，re.sub(pattern, repl, string)
    #\s匹配任意的空白符，+表示至少一个
    text = re.sub(r"上海交通大学(?:\s+|SHANGHAI JIAO TONG UNIVERSITY)+", "", text)
    text = re.sub(r"\s+\d+\s+", " ", text)  # 移除页码，\d表示任意的数字，用空格替换而不是直
                                            #接删除是防止前后拼接一起，内容混乱
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9.,;:()（）《》“”‘’\s-]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def parse_structured_content(documents):#传入一个列表，列表每一个元素是一个Document对象，包含page_content和metadata
    """解析文档为章节-条款结构"""
    full_text = "\n\n".join([doc.page_content for doc in documents])
    
    # 文档元数据提取
    doc_id = re.search(r"沪交教〔2023〕\d+号", full_text).group() if re.search(r"沪交教〔2023〕\d+号", full_text) else ""
    doc_title = "上海交通大学本科生管理规定" if "本科生管理规定" in full_text else "未知文档"
    category = "教学管理" if "管理规定" in doc_title else "其他"
    
    # 章节解析
    chapter_pattern = re.compile(r"(第[一二三四五六七八九十]+章\s+.*?)(?=第[一二三四五六七八九十]+章|$)", re.DOTALL)#章节之间划分的方式
    chapters = []
    
    for chapter_match in chapter_pattern.finditer(full_text):#遍历每一个章节
        chapter_text = chapter_match.group(1).strip()
        chapter_title = re.match(r"第[一二三四五六七八九十]+章\s+.*", chapter_text).group()
        
        # 条款解析
        clause_pattern = re.compile(r"(第[一二三四五六七八九十]+条\s+.*?)(?=第[一二三四五六七八九十]+条|$)", re.DOTALL)#条文之间划分的方式
        clauses = []
        
        for clause_match in clause_pattern.finditer(chapter_text):
            clause_text = clause_match.group(1).strip()
            clause_id = re.match(r"第[一二三四五六七八九十]+条", clause_text).group()
            
            # 标题提取逻辑
            clause_title = ""
            content_start = len(clause_id)
            
            # 检查是否存在明确的标题结构（如 "第XX条 [标题]：[内容]"）
            title_colon_pattern = re.compile(rf"{clause_id}\s+(.+?)：")
            title_colon_match = title_colon_pattern.search(clause_text)
            
            if title_colon_match:
                # 情况1：存在冒号分隔的标题
                clause_title = title_colon_match.group(1).strip()
                content_start = title_colon_match.end()  # 从冒号后开始
            else:
                # 情况2：检查是否存在"第XX条 [小标题] [内容]"结构（小标题后无冒号，但有明显换行或句号）
                title_pattern = re.compile(rf"{clause_id}\s+(.+?)(?=\n|。|$)")
                title_match = title_pattern.search(clause_text)
                
                if title_match and len(title_match.group(1)) < 20:  # 限制标题长度，避免误判
                    clause_title = title_match.group(1).strip()
                    content_start = title_match.end()
            
            # 提取条款内容
            clause_content = clause_text[content_start:].strip()
            
            # 解析子条款（如果存在）
            sub_items = []
            if "（一）" in clause_content:
                sub_pattern = re.compile(r"（[一二三四五六七八九十]+）(.*?)(?=（[一二三四五六七八九十]+）|$)", re.DOTALL)
                sub_items = []
                
                for sub_match in sub_pattern.finditer(clause_content):
                    sub_text = sub_match.group(1).strip()
                    sub_items.append(sub_text)
                
                # 清理主内容中的子条款标记
                clause_content = re.sub(r"（[一二三四五六七八九十]+）.*?(?=（[一二三四五六七八九十]+）|$)", "", clause_content, flags=re.DOTALL).strip()
            
            clauses.append({
                "clause_id": clause_id,
                "title": clause_title,
                "text": clause_content,
                "sub_items": sub_items if sub_items else None
            })
        
        chapters.append({
            "chapter": chapter_title,
            "clauses": clauses
        })
    
    return {
        "metadata": {
            "source": "SJTU2023版学生手册.pdf",
            "original_page_range": f"1-{len(documents)}",
            "category": category,
            "document_title": doc_title,
            "document_id": doc_id
        },
        "content": chapters
    }


def clean_json_newlines(data):
    """递归清理JSON数据中所有字符串的换行符"""
    if isinstance(data, dict):
        return {k: clean_json_newlines(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json_newlines(item) for item in data]
    elif isinstance(data, str):
        # 替换连续换行为单个空格（或直接删除）
        return re.sub(r"\n+", " ", data).strip()
    else:
        return data
    

def langchain_pdf_processor(pdf_path, output_json_path):
    """使用LangChain处理PDF并生成结构化JSON"""
    # 1. 加载PDF（LangChain的PyPDFLoader按页加载）
    loader = PyPDFLoader(pdf_path)#创建一个PyPDFLoader实例对象，传入PDF路径
    documents = loader.load()  # 返回List[Document]，每个Document含page_content和metadata（含页码）
    
    # 2. 清理每页文本
    cleaned_docs = []
    for doc in documents:
        cleaned_content = clean_text(doc.page_content)
        cleaned_docs.append(Document(
            page_content=cleaned_content,
            metadata=doc.metadata  
        ))#每页作为一个Document对象，包含清理后的内容和原始元数据
    
    # 3. 解析结构化内容
    structured_data = parse_structured_content(cleaned_docs)
    

    # 在保存JSON前调用
    structured_data = parse_structured_content(cleaned_docs)
    structured_data = clean_json_newlines(structured_data) 

    # 4. 保存为JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=2)
    print(f"LangChain处理完成,输出路径:{output_json_path}")


# 使用示例
if __name__ == "__main__":
    pdf_path = Path("SJTU2023版学生手册.pdf") 
    output_path = Path("langchain_structured_manual_4.json")
    langchain_pdf_processor(pdf_path, output_path)


