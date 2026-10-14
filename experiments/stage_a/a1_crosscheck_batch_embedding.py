"""A1 交叉校验：用与 Phase 15C.1 相同的批量 embed_documents 重算 Exact Top20，并与 exact20.json 比较。

用于证明线上逐条 embed_query 与 15C.1 的差异只来自批量编码的浮点差，而非 Exact 实现。
"""
import json,sys
sys.path[:0]=['D:/Agent_learn/evidenceFlow','D:/Agent_learn/evidenceFlow/src']
from evidenceflow.config import Settings
from evidenceflow.embedding.bge_m3 import BgeM3EmbeddingProvider
from evidenceflow.retrieval import ExactDenseIndex
from evidenceflow.storage.chroma import ChromaVectorStore
s=Settings(); st=ChromaVectorStore(s.vector_db_path,s.collection_name)
pairs=list(st.iter_all_chunks_with_embeddings()); idx=ExactDenseIndex([c for c,_ in pairs],[e for _,e in pairs])
q=json.load(open('evals/research_retrieval/research_questions_v1.json',encoding='utf-8'))['questions']
p=BgeM3EmbeddingProvider(s.embedding_model,device='cpu')
vs=p.embed_documents([x['english_query'] for x in q])
f=json.load(open('evals/evaluation_v3/phase15c1_exact_dense_architecture_v1/candidates/exact20.json',encoding='utf-8'))
by={}
for r in f['rows']: by.setdefault(r['question_id'],[]).append(r)
bad=0
for x,v in zip(q,vs):
    rs=sorted(by[x['question_id']],key=lambda r:r['candidate_rank'])
    res=idx.search(v,20)
    if [r['chunk_id'] for r in rs]!=[r.chunk.chunk_id for r in res] or [r['similarity_score'] for r in rs]!=[r.similarity_score for r in res]: bad+=1
print('batch-embedding mismatches:',bad)
