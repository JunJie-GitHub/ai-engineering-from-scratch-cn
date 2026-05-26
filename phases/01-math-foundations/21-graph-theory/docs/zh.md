# 机器学习中的图论

> 图（Graph）是表示关系的数据结构。如果你的数据包含连接关系，你就需要图论。

**类型：** 构建
**语言：** Python
**前置知识：** 第一阶段，第 01-03 课（线性代数、矩阵）
**预计时间：** 约 90 分钟

## 学习目标

- 构建一个图类，支持邻接矩阵（Adjacency Matrix）和邻接表（Adjacency List）表示，并实现广度优先搜索（Breadth-First Search, BFS）和深度优先搜索（Depth-First Search, DFS）遍历
- 计算图拉普拉斯矩阵（Graph Laplacian），并利用其特征值（Eigenvalues）检测连通分量（Connected Components）及对节点进行聚类
- 将一轮图神经网络（Graph Neural Network, GNN）风格的消息传递（Message Passing）实现为归一化邻接矩阵乘法
- 应用谱聚类（Spectral Clustering），利用菲德勒向量（Fiedler Vector）对图进行划分

## 问题背景

社交网络、分子结构、知识库、引文网络、道路地图——它们本质上都是图。传统机器学习（Machine Learning, ML）将数据视为扁平的表格，每一行相互独立，每一列代表一个特征。但当连接结构本身蕴含关键信息时，表格形式便无能为力了。

以社交网络为例。假设你想预测某位用户会购买什么商品，其自身的购买历史固然重要，但其好友的购买历史往往更具参考价值。连接关系本身承载着关键信号。

再以分子为例。假设你想预测它是否能与某种蛋白质结合，原子种类固然重要，但真正起决定作用的是原子之间的键合方式。在这里，结构即数据。

图神经网络（Graph Neural Networks, GNNs）是深度学习领域发展最快的方向之一。它们广泛应用于药物发现、社交推荐、欺诈检测以及知识图谱推理。而所有 GNN 都建立在同一个基础之上：基础图论。

你需要掌握以下四个核心要素：
1. 将图表示为矩阵的方法（以便进行矩阵乘法运算）
2. 用于探索图结构的遍历算法
3. 拉普拉斯矩阵（Laplacian）——谱图论（Spectral Graph Theory）中最重要的矩阵
4. 消息传递（Message Passing）——驱动 GNN 运行的核心操作

## 核心概念

### 图：节点与边（Graphs: Nodes and Edges）

图（Graph）$G = (V, E)$ 由顶点/节点（Node）$V$ 和边（Edge）$E$ 组成。每条边连接两个节点。

**有向图与无向图（Directed vs Undirected）。** 在无向图（Undirected Graph）中，边 $(u, v)$ 表示 $u$ 与 $v$ 相连，且 $v$ 也与 $u$ 相连。在有向图（Directed Graph / Digraph）中，边 $(u, v)$ 表示 $u$ 指向 $v$，但反之不一定成立。

**加权图与无权图（Weighted vs Unweighted）。** 在无权图（Unweighted Graph）中，边要么存在，要么不存在。在加权图（Weighted Graph）中，每条边都有一个数值权重（Weight）——例如距离、成本或连接强度。

| 图类型 | 示例 |
|-----------|---------|
| 无向无权图 | Facebook 好友关系网络 |
| 有向无权图 | Twitter 关注网络 |
| 无向加权图 | 道路地图（距离） |
| 有向加权图 | 网页链接（PageRank 分数） |

### 邻接矩阵（Adjacency Matrix）

邻接矩阵（Adjacency Matrix）$A$ 是图的核心表示形式。对于包含 $n$ 个节点的图：

A[i][j] = 1    if there is an edge from node i to node j
A[i][j] = 0    otherwise

对于无向图，$A$ 是对称矩阵：$A[i][j] = A[j][i]$。对于加权图，$A[i][j]$ 等于边 $(i, j)$ 的权重。

**示例——三角形图：**

Nodes: 0, 1, 2
Edges: (0,1), (1,2), (0,2)

A = [[0, 1, 1],
     [1, 0, 1],
     [1, 1, 0]]

邻接矩阵是每个图神经网络（Graph Neural Network, GNN）的输入。对 $A$ 进行的矩阵运算对应于对图本身的操作。

### 度（Degree）

节点的度（Degree）是指与其相连的边的数量。对于有向图，分为入度（In-degree，指向该节点的边）和出度（Out-degree，从该节点指出的边）。

度矩阵（Degree Matrix）$D$ 是一个对角矩阵：

D[i][i] = degree of node i
D[i][j] = 0    for i != j

以三角形图为例：$D = \text{diag}(2, 2, 2)$，因为每个节点都与其他两个节点相连。

度反映了节点的重要性。高度数节点通常充当枢纽节点（Hub Node）。网络的度分布（Degree Distribution）揭示了其整体结构。社交网络通常遵循幂律分布（Power Law，即少数枢纽节点和大量叶子节点）。而随机图（Random Graph）的度则服从泊松分布（Poisson Distribution）。

### 广度优先搜索与深度优先搜索（BFS and DFS）

这是两种基础的图遍历算法（Graph Traversal Algorithms），两者都不可或缺。

**广度优先搜索（Breadth-First Search, BFS）：** 优先探索所有直接邻居，然后再探索邻居的邻居。使用队列（Queue，先进先出 FIFO）实现。

BFS from node 0:
  Visit 0
  Queue: [1, 2]        (neighbors of 0)
  Visit 1
  Queue: [2, 3]        (add neighbors of 1)
  Visit 2
  Queue: [3]           (neighbors of 2 already visited)
  Visit 3
  Queue: []            (done)

BFS 用于在无权图中寻找最短路径。从起点到任意节点的距离，等于该节点在 BFS 中首次被发现时所在的层级。这也是 BFS 常用于计算社交网络中跳数距离（Hop-count Distance）的原因。

**深度优先搜索（Depth-First Search, DFS）：** 尽可能向深处探索，直到无法继续再回溯。使用栈（Stack，后进先出 LIFO）或递归实现。

DFS from node 0:
  Visit 0
  Stack: [1, 2]        (neighbors of 0)
  Visit 2               (pop from stack)
  Stack: [1, 3]         (add neighbors of 2)
  Visit 3               (pop from stack)
  Stack: [1]
  Visit 1               (pop from stack)
  Stack: []             (done)

DFS 适用于以下场景：
- 寻找连通分量（Connected Components）（从未访问的节点启动 DFS）
- 环检测（Cycle Detection）（通过 DFS 树中的回边判断）
- 拓扑排序（Topological Sorting）（按 DFS 完成时间的逆序排列）

| 算法 | 数据结构 | 查找目标 | 应用场景 |
|-----------|---------------|-------|----------|
| BFS | 队列（Queue） | 最短路径 | 社交网络距离计算、知识图谱遍历 |
| DFS | 栈（Stack） | 连通分量、环 | 连通性分析、拓扑排序 |

### 图拉普拉斯矩阵（Graph Laplacian）

$L = D - A$。这是谱图理论（Spectral Graph Theory）中最重要的矩阵。

以三角形图为例：

D = [[2, 0, 0],    A = [[0, 1, 1],    L = [[2, -1, -1],
     [0, 2, 0],         [1, 0, 1],         [-1, 2, -1],
     [0, 0, 2]]         [1, 1, 0]]         [-1, -1,  2]]

拉普拉斯矩阵具有以下显著特性：

1. **$L$ 是半正定矩阵（Positive Semi-definite）。** 其所有特征值（Eigenvalues）均大于等于 0。

2. **零特征值的数量等于连通分量的数量。** 连通图恰好有一个零特征值。包含 3 个不连通分量的图则有三个零特征值。

3. **最小的非零特征值（Fiedler 值）衡量图的连通性。** Fiedler 值越大，表示图连通性越好；Fiedler 值越小，表示图存在薄弱环节（即瓶颈）。

4. **Fiedler 值对应的特征向量（Fiedler 向量）揭示了最佳划分方式。** 特征向量值为正的节点归为一组，值为负的节点归为另一组。这就是谱聚类（Spectral Clustering）的原理。

graph TD
    subgraph "Graph to Matrices"
        G["Graph G"] --> A["Adjacency Matrix A"]
        G --> D["Degree Matrix D"]
        A --> L["Laplacian L = D - A"]
        D --> L
    end
    subgraph "Spectral Analysis"
        L --> E["Eigenvalues of L"]
        L --> V["Eigenvectors of L"]
        E --> C["Connected components (zeros)"]
        E --> F["Connectivity (Fiedler value)"]
        V --> S["Spectral clustering"]
    end

### 谱特性（Spectral Properties）

邻接矩阵和拉普拉斯矩阵的特征值无需任何遍历操作，即可揭示图的结构特性。

**谱聚类（Spectral Clustering）** 的工作原理如下：
1. 计算拉普拉斯矩阵 $L$
2. 找出 $L$ 的前 $k$ 个最小特征向量（Eigenvectors）（跳过第一个，因为对于连通图，它通常是全 1 向量）
3. 将这些特征向量作为每个节点的新坐标
4. 在这些新坐标上运行 K-Means 聚类算法

为什么这样做有效？$L$ 的特征向量编码了图上“最平滑”的函数。连接紧密的节点会获得相似的特征向量值，而被瓶颈隔开的节点则会获得差异较大的值。因此，特征向量能够自然地将不同的簇（Cluster）分离开来。

**与随机游走（Random Walk）的联系。** 归一化拉普拉斯矩阵（Normalized Laplacian）与图上的随机游走密切相关。随机游走的平稳分布（Stationary Distribution）与节点度成正比。混合时间（Mixing Time，即游走收敛的速度）则取决于谱隙（Spectral Gap）。

### 消息传递（Message Passing）

这是图神经网络（Graph Neural Network, GNN）的核心操作。每个节点收集来自其邻居的消息，进行聚合（Aggregate），然后更新自身的状态。

h_v^(k+1) = UPDATE(h_v^(k), AGGREGATE({h_u^(k) : u in neighbors(v)}))

在最简单的形式中，聚合操作取均值（Mean），更新操作为线性变换加激活函数：

h_v^(k+1) = sigma(W * mean({h_u^(k) : u in neighbors(v)}))

这本质上是矩阵乘法的变体。如果 $H$ 是所有节点特征的矩阵，$A$ 是邻接矩阵：

H^(k+1) = sigma(A_norm * H^(k) * W)

其中 $A_{\text{norm}}$ 是归一化邻接矩阵（Normalized Adjacency Matrix，每行之和为 1）。

一轮消息传递让每个节点“看到”其直接邻居。两轮传递使其能看到邻居的邻居。$K$ 轮传递则让每个节点获取其 $K$ 跳（K-hop）邻域内的信息。

graph LR
    subgraph "Round 0"
        A0["Node A: [1,0]"]
        B0["Node B: [0,1]"]
        C0["Node C: [1,1]"]
    end
    subgraph "Round 1 (aggregate neighbors)"
        A1["Node A: avg(B,C) = [0.5, 1.0]"]
        B1["Node B: avg(A,C) = [1.0, 0.5]"]
        C1["Node C: avg(A,B) = [0.5, 0.5]"]
    end
    A0 --> A1
    B0 --> A1
    C0 --> A1
    A0 --> B1
    C0 --> B1
    A0 --> C1
    B0 --> C1

### 核心概念与机器学习应用

| 概念 | 机器学习应用 |
|---------|---------------|
| 邻接矩阵（Adjacency Matrix） | GNN 输入表示 |
| 图拉普拉斯矩阵（Graph Laplacian） | 谱聚类、社区发现（Community Detection） |
| BFS/DFS | 知识图谱遍历、路径查找 |
| 度分布（Degree Distribution） | 节点重要性评估、特征工程 |
| 消息传递（Message Passing） | GNN 层（如 GCN、GAT、GraphSAGE） |
| $L$ 的特征值（Eigenvalues of L） | 社区发现、图划分（Graph Partitioning） |
| 谱聚类（Spectral Clustering） | 无监督节点分组 |
| PageRank | 节点重要性排序、网页搜索 |

## 构建

### 步骤 1：从零开始构建 Graph 类

class Graph:
    def __init__(self, n_nodes, directed=False):
        self.n = n_nodes
        self.directed = directed
        self.adj = {i: {} for i in range(n_nodes)}

    def add_edge(self, u, v, weight=1.0):
        self.adj[u][v] = weight
        if not self.directed:
            self.adj[v][u] = weight

    def neighbors(self, node):
        return list(self.adj[node].keys())

    def degree(self, node):
        return len(self.adj[node])

    def adjacency_matrix(self):
        import numpy as np
        A = np.zeros((self.n, self.n))
        for u in range(self.n):
            for v, w in self.adj[u].items():
                A[u][v] = w
        return A

    def degree_matrix(self):
        import numpy as np
        D = np.zeros((self.n, self.n))
        for i in range(self.n):
            D[i][i] = self.degree(i)
        return D

    def laplacian(self):
        return self.degree_matrix() - self.adjacency_matrix()

邻接表（Adjacency List）（`self.adj`）能够高效地存储邻居节点。转换为邻接矩阵（Adjacency Matrix）时使用了 `numpy`，因为后续所有的谱操作（Spectral Operations）都需要依赖它。

### 步骤 2：BFS 和 DFS

from collections import deque

def bfs(graph, start):
    visited = set()
    order = []
    distances = {}
    queue = deque([(start, 0)])
    visited.add(start)
    while queue:
        node, dist = queue.popleft()
        order.append(node)
        distances[node] = dist
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return order, distances


def dfs(graph, start):
    visited = set()
    order = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in reversed(graph.neighbors(node)):
            if neighbor not in visited:
                stack.append(neighbor)
    return order

BFS 使用双端队列（Deque）来实现 O(1) 的左侧弹出操作。DFS 则将列表作为栈（Stack）使用。两者均确保每个节点仅被访问一次，时间复杂度为 O(V + E)。

### 步骤 3：连通分量与拉普拉斯矩阵特征值

def connected_components(graph):
    visited = set()
    components = []
    for node in range(graph.n):
        if node not in visited:
            order, _ = bfs(graph, node)
            visited.update(order)
            components.append(order)
    return components


def laplacian_eigenvalues(graph):
    import numpy as np
    L = graph.laplacian()
    eigenvalues = np.linalg.eigvalsh(L)
    return eigenvalues

`eigvalsh` 专用于对称矩阵（Symmetric Matrix）——对于无向图而言，拉普拉斯矩阵（Laplacian Matrix）始终是对称的。该函数会按升序返回特征值（Eigenvalues）。通过统计零特征值的数量，即可确定连通分量（Connected Components）的个数。

### 步骤 4：谱聚类

def spectral_clustering(graph, k=2):
    import numpy as np
    L = graph.laplacian()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    features = eigenvectors[:, 1:k+1]

    labels = np.zeros(graph.n, dtype=int)
    for i in range(graph.n):
        if features[i, 0] >= 0:
            labels[i] = 0
        else:
            labels[i] = 1
    return labels

当 k=2 时，菲德勒向量（Fiedler Vector）的符号可将图划分为两个簇（Cluster）。当 k>2 时，则需在前 k 个特征向量（Eigenvectors）上运行 K-Means 算法（需排除全为 1 的平凡特征向量）。

### 步骤 5：消息传递

def message_passing(graph, features, weight_matrix):
    import numpy as np
    A = graph.adjacency_matrix()
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A_norm = A / row_sums
    aggregated = A_norm @ features
    output = aggregated @ weight_matrix
    return output

这代表了图神经网络（Graph Neural Network, GNN）中单轮的消息传递（Message Passing）过程。每个节点的新特征是其邻居节点特征的加权平均值，并经过权重矩阵（Weight Matrix）的线性变换。通过堆叠多轮传递，可实现信息的进一步传播。

## 实际应用

借助 `networkx` 和 `numpy`，相同的操作只需一行代码即可完成：

import networkx as nx
import numpy as np

G = nx.karate_club_graph()

A = nx.adjacency_matrix(G).toarray()
L = nx.laplacian_matrix(G).toarray()

eigenvalues = np.linalg.eigvalsh(L.astype(float))
print(f"Smallest eigenvalues: {eigenvalues[:5]}")
print(f"Connected components: {nx.number_connected_components(G)}")

communities = nx.community.greedy_modularity_communities(G)
print(f"Communities found: {len(communities)}")

pr = nx.pagerank(G)
top_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"Top 5 PageRank nodes: {top_nodes}")

`networkx` 凭借优化的 C 语言后端，能够处理任意规模的图。在生产环境中请直接使用它。而你从零开始实现的代码，则用于深入理解其底层原理。

### 基于 numpy 的谱分析（spectral analysis）

import numpy as np

A = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0]
])

D = np.diag(A.sum(axis=1))
L = D - A

eigenvalues, eigenvectors = np.linalg.eigh(L)
print(f"Eigenvalues: {np.round(eigenvalues, 4)}")
print(f"Fiedler value: {eigenvalues[1]:.4f}")
print(f"Fiedler vector: {np.round(eigenvectors[:, 1], 4)}")

fiedler = eigenvectors[:, 1]
group_a = np.where(fiedler >= 0)[0]
group_b = np.where(fiedler < 0)[0]
print(f"Cluster A: {group_a}")
print(f"Cluster B: {group_b}")

菲德勒向量（Fiedler vector）承担了核心计算工作。其中一个簇的对应分量为正值，另一个簇则为负值。整个过程无需迭代优化——仅需一次特征分解（eigendecomposition）即可。

## 交付成果

本课时将产出：
- `outputs/skill-graph-analysis.md` —— 用于分析图结构数据（graph-structured data）的技能参考指南

## 概念关联

| 概念 | 应用场景 |
|---------|------------------|
| 邻接矩阵（adjacency matrix） | GCN、GAT、GraphSAGE 的输入 |
| 拉普拉斯矩阵（Laplacian） | 谱聚类（spectral clustering）、ChebNet 滤波器 |
| 广度优先搜索（BFS） | 知识图谱遍历、最短路径查询 |
| 消息传递（message passing） | 每个 GNN 层、神经消息传递 |
| 谱间隙（spectral gap） | 图连通性、随机游走的混合时间 |
| 度分布（degree distribution） | 幂律网络、节点特征工程 |
| 连通分量（connected components） | 数据预处理、处理非连通图 |
| PageRank 算法 | 节点重要性排序、注意力机制初始化 |

图神经网络（GNN）值得特别提及。GCN（Kipf & Welling, 2017）中的图卷积操作使用了添加自环（self-loops）的邻接矩阵，即 `A_hat = A + I`：

H^(l+1) = sigma(D_hat^(-1/2) * A_hat * D_hat^(-1/2) * H^(l) * W^(l))

其中 `A_hat = A + I`（邻接矩阵加上自环），`D_hat` 是 `A_hat` 的度矩阵（degree matrix）。自环确保了每个节点在聚合过程中能够包含自身的特征。这正是采用对称归一化（symmetric normalization）的消息传递机制。`D_hat^(-1/2) * A_hat * D_hat^(-1/2)` 即为归一化邻接矩阵。拉普拉斯矩阵在此出现，是因为该归一化操作与 `L_sym = I - D^(-1/2) * A * D^(-1/2)` 密切相关。理解拉普拉斯矩阵，也就理解了 GCN 为何有效。

## 练习

1. **从零实现网页排名算法（PageRank）**。初始分数设为均匀分布。每一步更新公式为：`score(v) = (1-d)/n + d * sum(score(u)/out_degree(u))`，其中 `u` 指向 `v`。设置阻尼系数 `d=0.85`。迭代直至收敛（变化量 < 1e-6）。在小型网络图上进行测试。

2. **使用谱聚类（Spectral Clustering）发现社区**。构建一个包含两个明显分离簇的图（例如，两个通过单条边连接的团 (Clique)）。运行谱聚类算法，验证其是否能正确划分。随着跨簇边的增加，结果会发生什么变化？

3. **实现 Dijkstra 算法（迪杰斯特拉算法）**，用于求解加权图中的最短路径。在相同图结构且权重均匀的情况下，将其结果与广度优先搜索（BFS）进行对比。

4. **构建一个双层消息传递网络（Message Passing Network）**。使用不同的权重矩阵执行两次消息传递。证明经过两轮迭代后，每个节点都聚合了其 2 跳邻域（2-hop Neighborhood）内的信息。

5. **分析真实世界图数据**。使用空手道俱乐部图（Karate Club Graph，34 个节点，78 条边）。计算度分布（Degree Distribution）、拉普拉斯矩阵（Laplacian Matrix）特征值，并执行谱聚类。将谱聚类结果与已知的真实划分（Ground Truth）进行对比。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 图（Graph） | “节点和边” | 一种数学结构 G=(V,E)，用于编码成对关系 |
| 邻接矩阵（Adjacency Matrix） | “连接表” | 一个 n x n 矩阵，若节点 i 和 j 相连，则 A[i][j] = 1 |
| 度（Degree） | “节点的连接程度” | 与节点相连的边的数量 |
| 拉普拉斯矩阵（Laplacian） | “D 减 A” | L = D - A，其特征值可揭示图结构的矩阵 |
| 菲德勒值（Fiedler Value） | “代数连通度” | L 的最小非零特征值，用于衡量图的连通紧密程度 |
| 广度优先搜索（BFS） | “逐层搜索” | 一种遍历算法，在深入下一层前访问所有邻居节点，用于寻找最短路径 |
| 深度优先搜索（DFS） | “优先深入” | 一种遍历算法，沿一条路径走到底后再回溯 |
| 消息传递（Message Passing） | “节点与邻居通信” | 每个节点聚合其邻居的信息，是图神经网络（GNN）的核心机制 |
| 谱聚类（Spectral Clustering） | “按特征向量聚类” | 利用拉普拉斯矩阵的特征向量对图进行划分 |
| 连通分量（Connected Component） | “独立的一块” | 一个极大子图，其中任意节点均可到达其他所有节点 |

## 延伸阅读

- **Kipf & Welling (2017)** ——《Semi-Supervised Classification with Graph Convolutional Networks》。这篇论文开启了现代图神经网络（Graph Neural Networks, GNN）的研究。文中展示了谱图卷积（Spectral Graph Convolution）如何简化为消息传递机制。
- **Spielman (2012)** ——《Spectral Graph Theory》讲义。关于拉普拉斯矩阵、谱间隙（Spectral Gap）和图划分（Graph Partitioning）的权威入门资料。
- **Hamilton (2020)** ——《Graph Representation Learning》。全面介绍图神经网络从基础理论到实际应用的专著。
- **Bronstein et al. (2021)** ——《Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges》。提出统一理论框架的奠基性论文。
- **Veličković et al. (2018)** ——《Graph Attention Networks》。将注意力机制（Attention Mechanism）引入消息传递过程的扩展研究。