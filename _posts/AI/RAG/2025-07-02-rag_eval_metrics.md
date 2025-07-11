---
title: "RAG eval metrics"
categories: [AI, RAG]
tags: [AI, RAG]
author: JiHyun Kim
date: 2025-07-01
hide: false
---


회의록 기반 Task분리 + wiki 문맥 기반 RAG 추천 시스템이므로
기존 논문용 QA 성능 지표를 그대로 쓰기보단, 서비스 맥락에 맞춘 정성/정량 혼합 평가가 필요함

1. 역할에 맞는 테스크가 추출되었나
- role-aware prompt 
- wiki문맥의 반영 정도 및 포지션 적합성 확인 -> 기존에는 포지션에 맞는 main task로 잘 분리되지 못하였음.
<전 -> 후 에 맞게끔 성능 개선된걸 보여줄 예정>


2. 문맥에 맞는, 유의미하고 적절한 subtask를 추천하고 있는가



<정량적 평가>
1. 정확도 유사도: task 추천 결과가 실제 wiki 문맥과 얼마나 유사한가?
- KoSimCSE 등 임베딩 기반 cosine similarity 

2. 중복률 감소: 동일한 회의에서 중복되는 task를 반복 생성하지 않는가? 
- 기존 task embedding들과의 거리 기반 중복 판단

3. Latency: 추천까지 걸리는 시간 
- FastAPI + time.time() 측정


~<정성적 평가 - HITL 기반 수동 점검>~ 
<정성적 평가 - LLM as a Judge>
- Pointwise
- Pairwise
- Listwise

관련성
해당 task는 회의 내용에 적절히 반응했는가?
✔️/❌ 또는 5점 척도

역할 적합성
추천된 task가 해당 포지션(기획/프론트/백엔드 등)에 적절한가?
팀원 피드백 수집

명확성
task가 구체적이고 actionable한가?
1~5점 평가


<평가 측정은>
- rag로 시험을 보면 되지않니 ? 