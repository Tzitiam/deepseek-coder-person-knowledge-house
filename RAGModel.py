import chromadb
from typing import List
from sentence_transformers import SentenceTransformer
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ 自定义嵌入函数（兼容 ChromaDB 1.0+）
class BGEZhEmbeddingFunction:
    

    def __init__(self, model_path: str = "./model_path/bge-small-zh-v1.5"):
        try:
            logger.info("正在加载Embedding模型...")
            self.model = SentenceTransformer(model_path)
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def __call__(self, input: List[str]) -> List[List[float]]:
        # 确保返回 List[List[float]] 格式
        try:
            embeddings = self.model.encode(input, normalize_embeddings=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding计算失败: {e}")
            raise


if '__name__'=='__main__':
    # 初始化 ChromaDB 客户端
    client = chromadb.PersistentClient(path="./chroma_db")

    # ✅ 创建集合（绑定 BGE 中文嵌入模型）
    collection = client.get_or_create_collection(
        name="bge_zh_collection",
        embedding_function=BGEZhEmbeddingFunction()  # 传入自定义嵌入函数
    )

    # 测试插入数据
    # collection.add(
    #     documents=["深度求索（DeepSeek）是一家AI公司", "ChromaDB 用于向量检索"],
    #     ids=["doc1", "doc2"]
    # )

    # 查询测试
    results = collection.query(
        query_texts=["chromadb是做什么的？"],
        n_results=1
    )
    print(results)
