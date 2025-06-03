import os
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import logging
from RAGModel import BGEZhEmbeddingFunction

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBuilder:
    def __init__(self):
        self.docs_dir = "./knowledge_files"
        self.chroma_dir = "./chroma_db"
        
        # # 初始化模型（增加错误处理）
        # try:
        #     logger.info("正在加载Embedding模型...")
        #     self.embedding_model = SentenceTransformer(
        #         "./model_path/bge-small-zh-v1.5",
        #         device="cpu",
        #         cache_folder="./model_cache"  # 指定缓存目录
        #     )
        #     logger.info("模型加载成功")
        # except Exception as e:
        #     logger.error(f"模型加载失败: {e}")
        #     raise
        
        # 初始化ChromaDB
        self.client = chromadb.PersistentClient(path=self.chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=BGEZhEmbeddingFunction()
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
    
    # def _calculate_embeddings(self, texts):
    #     """计算embedding并增加错误处理"""
    #     try:
    #         return self.embedding_model.encode(
    #             texts,
    #             normalize_embeddings=True,
    #             show_progress_bar=True
    #         ).tolist()
    #     except Exception as e:
    #         logger.error(f"Embedding计算失败: {e}")
    #         raise
    
    def build(self):
        """构建知识库"""
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir)
            logger.warning(f"创建了空文档目录: {self.docs_dir}")
            return

        documents = []
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                if file.endswith((".txt", ".md", ".pdf")):
                    path = os.path.join(root, file)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            text = f.read()
                            chunks = self.text_splitter.split_text(text)
                            documents.extend(chunks)
                            logger.info(f"已处理: {file} -> {len(chunks)}个片段")
                    except UnicodeDecodeError:
                        logger.warning(f"跳过非文本文件: {file}")
                    except Exception as e:
                        logger.error(f"处理文件{file}出错: {e}")
        
        if not documents:
            logger.warning("未找到可处理的文档！")
            return
        
        try:
            # 分批处理避免内存溢出
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                print('--------------------')
                batch = documents[i:i + batch_size]
                print(batch)
                print('--------------------')
                embeddings = BGEZhEmbeddingFunction().__call__(batch)
                print('--------------------')
                self.collection.add(
                    ids=[f"doc_{i+j}" for j in range(len(batch))],
                    documents=batch,
                    embeddings=embeddings,
                    metadatas=[{"source": "file"}] * len(batch)
                )
                logger.info(f"已入库批次 {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            logger.info(f"知识库构建完成，共 {len(documents)} 个文本片段")
        except Exception as e:
            logger.error(f"构建知识库失败: {e}")
            raise

if __name__ == "__main__":
    try:
        builder = KnowledgeBuilder()
        builder.build()
    except Exception as e:
        logger.critical(f"程序异常终止: {e}")
