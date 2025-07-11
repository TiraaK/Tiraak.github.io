---
title: "Tavily APIs"
categories: [Engineering, API]
tags: [Crawling, Research, Deep Dive]
date: 2025-07-11
hide: false
---
⏺ **payload**는 Tavily API에 보내는 검색 요청 데이터입니다:

```python
  payload = {
      "api_key": self.tavily_api_key,        # API 인증 키
      "query": query,                        # 검색할 쿼리
      "search_depth": "advanced",            # 검색 깊이 
  (advanced/basic)
      "include_answer": True,                # AI 답변 포함 여부
      "include_domains": [                   # 검색할 도메인 
  제한
          "tistory.com",
          "velog.io",
          "brunch.co.kr",
          "blog.naver.com",
          "inflearn.com"
      ],
      "max_results": max_results,            # 최대 결과 수
      "country": "kr"                        # 검색 국가
  }

```