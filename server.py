from enum import Enum
from typing import List, Tuple
from FlagEmbedding import BGEM3FlagModel
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
import os


# gpu batch_size in order of your available vram
batch_size = 12 if os.getenv('BGE_M3_BATCH_SIZE') == "" or os.getenv('BGE_M3_BATCH_SIZE') == None else int(os.getenv('BGE_M3_BATCH_SIZE'))
# max context length for embeddings and passages in re-ranker
max_length = 8192 if os.getenv('BGE_M3_MAX_LENGTH') == "" or os.getenv('BGE_M3_MAX_LENGTH') == None else int(os.getenv('BGE_M3_MAX_LENGTH'))
# max context length for questions in re-ranker
max_query_length = 512 if os.getenv('BGE_M3_MAX_QUERY_LENGTH') == "" or os.getenv('BGE_M3_MAX_QUERY_LENGTH') == None else int(os.getenv('BGE_M3_MAX_QUERY_LENGTH'))
# re-rank score weights
rerank_weights = [0.4, 0.2, 0.4] if os.getenv('BGE_M3_RERANKER_WEIGHTS') == "" or os.getenv('BGE_M3_RERANKER_WEIGHTS') == None else [float(x) for x in os.getenv('BGE_M3_RERANKER_WEIGHTS').split(",")]
model_name = 'BAAI/bge-m3' if os.getenv('BGE_M3_MODEL_NAME') == "" or os.getenv('BGE_M3_MODEL_NAME') == None else os.getenv('BGE_M3_MODEL_NAME')


class EmbeddingType(Enum):
    dense = 'dense'
    sparse = 'sparse'
    colbert = 'colbert'


class m3Wrapper:
    def __init__(self, model_name: str):
        self.model = BGEM3FlagModel(model_name, use_fp16=True)

    def embedding(self, sentences: List[str], type: EmbeddingType) -> List[List[float]]:
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True if type == None or type == EmbeddingType.dense else False,
            return_sparse=True if type == EmbeddingType.sparse else False,
            return_colbert_vecs=True if type == EmbeddingType.colbert else False,
        )
        if embeddings['dense_vecs'] is not None:
            return embeddings['dense_vecs'].tolist()
        else:
            return []

    def reranker(self, sentences: List[Tuple[str, str]]) -> List[float]:
        scores = self.model.compute_score(
            sentences,
            batch_size=batch_size,
            max_query_length=max_query_length,
            max_passage_length=max_length,
            weights_for_different_modes=rerank_weights,
        )
        return scores['colbert+sparse+dense']


model = m3Wrapper(model_name)


class EmbeddingResponse(BaseModel):
    vectors: List[List[float]]


class EmbeddingRequest(BaseModel):
    sentences: List[str] = Field(
        title="The reranker sentences", max_length=8192, max_items=10
    )
    type: EmbeddingType | None


class RerankerRequest(BaseModel):
    target: str = Field(
        title="The reranker target string", max_length=8192
    )
    sentences: List[str] = Field(
        title="The reranker sentences", max_length=8192, max_items=10
    )


class RerankerResponse(BaseModel):
    scores: List[float]


app = FastAPI()


@app.post("/embedding", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    return EmbeddingResponse(vectors=model.embedding(request.sentences, request.type))


@app.post("/reranker", response_model=RerankerResponse)
async def rerank(request: RerankerRequest):
    sentence_pars = []
    for s in request.sentences:
        sentence_pars.append([request.target, s])
    return RerankerResponse(scores=model.reranker(sentence_pars))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
# curl -X POST -k -v http://127.0.0.1:3000/embedding -H "Content-Type: application/json" -d '{"sentences":["xx"], "type":"dense"}'
# curl -X POST -k -v http://127.0.0.1:3000/reranker -H "Content-Type: application/json" -d '{"target": "What is BGE M3?","sentences":["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", "xxx"], "type":"dense"}'
