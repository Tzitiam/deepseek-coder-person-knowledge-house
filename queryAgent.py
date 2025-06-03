from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict
import torch
import os
from RAGModel import BGEZhEmbeddingFunction

os.environ["CHROMA_DISABLE_DEFAULT_MODEL"] = "true"  #  必须在创建客户端前设置
BGE_MODEL="./model_path/bge-small-zh-v1.5"
DEEPSEEK_CODER_MODEL_PATH = './model_path/deepseek-coder-1.3b-base'
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "knowledge_base"

class IntelligentQA:
    def __init__(self):
        
        # 初始化DeepSeek生成模型
        self.generator_tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_CODER_MODEL_PATH)
        self.generator_model = AutoModelForCausalLM.from_pretrained(
            DEEPSEEK_CODER_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
    def select(self,query="第一件事是什么？"):
        client = chromadb.PersistentClient(path="./chroma_db")

        # ✅ 访问集合（绑定 BGE 中文嵌入模型）
        collection = client.get_collection(
                name="knowledge_base",
                embedding_function=BGEZhEmbeddingFunction()  # 传入自定义嵌入函数
            )

            # 测试插入数据
            # collection.add(
            #     documents=["深度求索（DeepSeek）是一家AI公司", "ChromaDB 用于向量检索"],
            #     ids=["doc1", "doc2"]
            # )

            # 查询测试
        results = collection.query(
                query_texts=list(query),
                n_results=3,
        )
        return results
    def retrieve(self, query: str) -> List[Dict]:
        """BGE检索核心方法"""
        results = self.select(query)
        return [
            {"content": doc, "metadata": meta or {}}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

    def generate_response(self, question: str, contexts: List[str]) -> str:
        """DeepSeek生成优化回答"""
        context_str = "\n".join([f"[参考{i+1}]: {ctx['content']}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""你是一个专业的技术助手，请根据以下参考内容回答问题。
        
                    [问题]: {question}

                    [参考内容]:
                    {context_str}

                    请按照以下要求生成回答：
                    1. 直接解决问题，不要复述问题
                    2. 如果参考内容存在冲突，指出差异并提供建议
                    3. 包含示例代码时添加详细注释
                    4. 使用中文回答

                    [最终答案]:"""
        
        inputs = self.generator_tokenizer(prompt, return_tensors="pt").to(self.generator_model.device)
        outputs = self.generator_model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True
        )
        return self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def query(self, question: str) -> str:
        """端到端问答流程"""
        # 第一阶段：检索
        contexts = self.retrieve(question)
        if not contexts:
            return "未找到相关参考资料，请尝试其他提问方式"
        
        # 第二阶段：生成
        return self.generate_response(question, contexts)

# 使用示例
if __name__ == "__main__":
    qa_system = IntelligentQA()
    
    # 交互式问答
    print("技术问答系统已就绪（输入q退出）")
    while True:
        question = input("\n您的问题：").strip()
        if question.lower() == 'q':
            break
            
        response = qa_system.query(question)
        print("\nAI回答：")
        print(response)
