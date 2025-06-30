---
title: "Python의 stack,queue,graph"
categories: [Algorithm, Python]
tags: [Algorithm ,Python, Data Structure]
date: 2019-12-17 2:50:00 +0900
hide: false
---

Algorithm은 time complexity를 사용하여 성능을 비교한다.
이제부터 stack, queue에 대한 구현을 time complexity를 사용하여 성능 비교해 보겠다.

## Stack in Python
stack 을list로 구현
```python
    import time
    a= [i for i in range(50000)]

    # 얘는 stack- stack은 list로 구현
    b= time.time()
    for i in range(10000):
        a.append(1)
        a.pop()
    print('stack ', time.time()-b)
```
## queue in Python
stack은 list를 사용하여 구현 가능했는데
그렇다면 queue도 stack과 같이 list로 구현해볼 수 있을까?
```python
b = time.time()
for i in range(10000):
    a.append(1)
    del a[0]
print('queue 맨 뒤 추가후 맨 앞 삭제 ',time.time() - b)



b = time.time()
for i in range(10000):
    a.insert(0,1)
    a.pop()
print('queue 맨 앞 추가후 맨 뒤 삭제 ',time.time() - b)
```

***
```python
#결과: 
#stack: 0.002447843551635742
#queue 맨 뒤 추가후 맨 앞 삭제: 0.15894508361816406
#queue 맨 앞 추가후 맨 뒤 삭제: 0.22293305397033691

    # --> 즉, queue에서 list사용하면 시간 엄청 걸린다
    # --> 따라서 큐는 파이썬에서 제공하는 collections.deque()를 시용한다.
```
***

```python
# append() appendleft() pop() popleft()로 함수가 원하는 방향에서 데이터 넣기,빼기 가능
import collections
deq = collections.deque(a)
b = time.time()

for i in range(10000):
    deq.append(1)
    deq.popleft()
print(time.time() - b)

for i in range(10000):
    deq.appendleft(1)
    deq.pop()
print(time.time() - b)
```

# graph - 여러 물체간의 관계를 set을 이용해서 node, edge로 표현하고
# 추가,삭제 하려면 add(),remove()를 가진 함수를 가진 클래스를 생성해야한다.
```python
g = {1: {2,5},
     2: {1,3,5},
     3: {2,4},
     4: {3,5,6},
     5: {1,2,4},
     6: {4}}

class graph:
    def __init__(self):
        self.data = {}
    def add_node(self,n):
        self.data[n] = set()
    def del_node(self,n):
        for i in self.data[n]:    #5라는 node를 지우기 전에 1,2,4가 가지고 있는 5를 지워줘야 한다.
            self.data[i].remove(n)
        del self.data[n]
    def add_edge(self,n,m):
        self.data[n].add(m) #n입장에서는 m이 이웃
        self.data[m].add(n) #m입장에서는 n이 이웃
    def del_edge(self,n,m): #n 과 m이 서로 연결되어잇는 edge는
        self.data[n].remove(m) #n에서 m을 지우고
        self.data[m].remove(n) #m에서 n을 지운다

```
```python
#graph탐색- DFS : 시작 노드에서 갈 수 있을때까지 deep하게 가다가 더이상 진행이 불가능할 경우 뒤로 돌아가서 다음 방향을 탐색
                # stack을 이용하는 대표적인 알고리즘!
def dfs(self,n):
    chk = [False]*(len(self.data)+1)  #일단 모두 방문 안한 상태로 냅둔다
    stack = []
    stack.append(n)

    while stack:
        cur = stack.pop()
        if chk[cur]==False:
            print(cur)
        chk[cur] = True

        for i in self.data[cur]:
            if chk[i]==False:
                stack.append(i)
#grpah탐색- BFS:시작 노드에서 방문할 수 있는 모든 노드를 방문
def bfs(self,n):
    import collections
    chk = [False]*(len(self.data)+1)  #일단 모두 방문 안한 상태로 냅둔다
    q = collections.deque()
    q.append(n)
    while q:
        cur = q.popleft()

        if chk[cur]==False:
            print(cur,end=' ')
        chk[cur] = True

        for i in self.data[cur]:
            if chk[i]==False:
                q.append(i)
    print()
```

# 이제 graph가 무엇인지도 보고, graph 탐색도 해봤겠다
# tree구조와 heap구조(rooted tree)를 알아보자
# tree구조: graph중 cycle이 없는것
# min heap/max heap: tree중에서도 rooted tree처럼 표시 하는 것 (rooted tree)
```python
#min heap: min heap을 사용하면 원소들이 항상 정렬된 상태로 추가되고 삭제되며, min heap에서 가장 작은값은 언제나 인덱스 0, 즉, 이진 트리의 루트에 위치합니다.
# 내부적으로 min heap 내의 모든 원소(k)는 항상 자식 원소들(2k+1, 2k+2) 보다 크기가 작거나 같도록 원소가 추가되고 삭제됩니다.
import heapq
a=[1,6,4,3,7,9]
heapq.heapify(a)
print(a)
heapq.heappush(a,-3)
print(a)
print(heapq.heappop(a))
print(heapq.heappop(a))
print(heapq.heappop(a))
```