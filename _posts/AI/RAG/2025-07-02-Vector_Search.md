---
title: "RAG eval metrics"
categories: [AI, RAG]
tags: [AI, RAG]
author: JiHyun Kim
date: 2025-07-01
hide: false
---


## 유사 KG
main task: role embedding으로 분류
subtask: 기존 wiki vdb 활용하되, role임베딩으로 강화



## Hybrid Appoach
config에서 어떤 method 쓸건지 설정

```python
Class HybridApproach:
    def __init__(self):
        self.vector_fusion = VectorFusionRetriever()
        self.cls_based = CLSTokenRetriever()
        self.kg_based = KGRetriever()



    HybridApproach[config.ModelSelector]

    

    def retriever_wiki_context(self, task, project_id, role, k=1):
        fusion_result = self.vector_fusion.retrieve(task, project_id)
```

### 방법1: Basic RAG
* 흐름: 단순 쿼리 -> 벡터 -> 검색
* ex) task mapping 


### 방법2: Query Enhancement 방식
* 흐름: 쿼리변형 -> 벡터 -> 검색
* ex) CLS Token concat


- 명시적 관계성: blackbox
- 논리적추론: Transformer모델 내부 처리
- Explanabiity : 없음
- 구조화: 텍스트 연결로 작업 진행
- 증거기반: 근거 추적 불가


### 방법3: Vector Fusion 방식
* 흐름: 다중 벡터 결합 -> 검색
* ex) Role + Task 임베딩 결합


- 명시적 관계성: 벡터 연산으로 관계성 사라짐 
- 논리적 추론: 수치계산에 국한됨 
- Explanabiity: 가중치는 알 수 있음 
- 구조화: Vector Space만 존재 
- 증거기반: 유사도 점수만 존재 


```python
Class RoleEnhancedWR:
    def __init__(self):
        self.wiki_retriever = WikiRetriever()
        self.role_embeddings = {
            "AI": self._create_ai_seed_embedding(),
            "BE": self._create_be_seed_embedding(), 
            "FE": self._create_fe_seed_embedding(),
            "CLOUD": self._create_cloud_seed_embedding()
        }
        self.embedder = CustomEmbeddingFunction()
```


### 방법4: Level 4: Multi-Vector Systems 





### 방법5: KG



- 명시적 관계성: 엔티티 간 관계 명시
- 논리적 추론: 그래프 순회로 추론
- Explanabiity: 추론 경로 추적 가능
- 구조화: 체계적 지식 구조
- 증거기반: 관계 기반 증거 제시 

단점: 구축비용이 엄청나다..
---

1 Simple Query: SELECT * WHEREvector_db.search()⭐
2 Query Enhancement: WHERE (A OR B OR C)embed(enhanced_query)⭐⭐
3 Vector Fusion: JOIN + 가중치 계산가중 벡터 결합⭐⭐⭐
3.5 Smart Context Filtering: 
4파티션 + UNION ALLMulti-DB + 라우팅⭐⭐⭐⭐
5저장프로시저 + 룰엔진Vector + Symbol 결합⭐⭐⭐⭐⭐