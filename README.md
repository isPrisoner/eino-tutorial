# Eino Tutorial RAG 项目

基于 Eino 框架的工程级 RAG（检索增强生成）系统，集成了 Milvus 向量数据库和火山引擎大模型。

## ⭐ 项目亮点

- **工程级 RAG 实现**：非 demo 级别，完成从自定义架构到 Eino 标准接口的完整重构
- **标准接口重构**：从自定义 VectorStore 重构为 Eino SchemaDocumentWriter/Reader，提升可维护性
- **三层架构设计**：mapper/repository/store 分层，职责清晰，易于扩展和测试
- **统一文档模型**：基于 Eino schema.Document，实现业务层与存储层的解耦
- **高可扩展性**：支持多种检索策略扩展（向量检索、混合检索、重排序）

## 项目概述

本项目展示了如何使用 Eino 框架构建一个完整的 RAG 系统，包括：

- 文档入库：支持文本、文件、目录导入
- 向量存储：使用 Milvus 向量数据库
- 文档检索：基于向量相似度的语义检索
- 问答生成：结合检索结果生成答案

## 快速开始

### 环境要求

- Go 1.21+
- Milvus 向量数据库（使用 Docker 启动）
- 火山引擎 API Key

### 启动 Milvus

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest
```

### 环境变量配置

```bash
# 火山引擎 API 配置
export ARK_API_KEY=your_ark_api_key
export ARK_MODEL_NAME=doubao-pro-32k
export EMBEDDER=doubao-embedding-vision-251215
export ARK_EMBEDDER_API_KEY=your_ark_api_key

# Milvus 配置（可选，有默认值）
export MILVUS_ADDRESS=127.0.0.1:19530
export MILVUS_DIMENSION=2048
export MILVUS_TOPK=3

# 文本切分配置（可选，有默认值）
export CHUNK_SIZE=500
export CHUNK_OVERLAP=50

# RAG 配置（可选，有默认值）
export RAG_MIN_SCORE=0.5
export RAG_TOPK=5
export RAG_MAX_CONTEXT_LEN=2000
export RAG_MAX_CONTEXT_CHUNKS=10
```

### 运行程序

```bash
go run cmd/main.go
```

## CLI 使用方式

### 添加文档

```bash
/add 这是要添加到知识库的文档内容
```

### 添加文件

```bash
/add_file ./data/document.txt
```

### 添加目录

```bash
/add_dir ./data/
```

### RAG 查询

```bash
/rag 你的问题
```

### 其他命令

- `/translate 翻译内容` - 翻译文本
- `/code 代码生成需求` - 生成代码
- `/summarize 文本内容` - 总结文本
- 直接输入其他内容进行普通对话

## 架构说明

### 核心组件

```
internal/
├── vectorstore/          # 向量存储
│   ├── interface.go      # SchemaDocumentWriter/Reader 接口
│   └── milvus/          # Milvus 实现
│       ├── milvus.go    # MilvusStore 实现
│       ├── mapper.go    # schema.Document ↔ MilvusRow 转换
│       ├── repository.go # Milvus 数据访问层
│       └── collection.go # Collection 管理
├── ingest/              # 文档入库服务
│   └── service.go       # 实现 Eino Indexer 接口
├── retrieval/           # 文档检索服务
│   └── service.go       # 实现 Eino Retriever 接口
├── chat/                # 聊天机器人
│   └── chatbot.go       # ChatBot 实现
├── cli/                 # CLI 命令处理
│   ├── handler.go       # 命令处理器
│   └── commands.go      # 命令定义
├── docconv/             # 文档转换
│   └── converter.go     # BuildSchemaDocument 辅助函数
├── textsplitter/        # 文本切分
│   └── textsplitter.go  # 文本切分实现
├── fileimport/          # 文件导入
│   └── fileimport.go    # 文件读取和目录扫描
└── utils/               # 工具函数
    ├── log.go          # 日志工具
    └── vector.go       # 向量类型转换
```

### 🧠 架构设计亮点

#### 为什么从 VectorStore 重构为 SchemaDocumentWriter/Reader

原架构使用自定义的 VectorStore 接口和 Document 结构，导致：
- 与 Eino 生态不兼容，无法复用 Eino 组件
- 业务层与存储层强耦合，难以替换向量数据库
- 接口过于庞大（包含插入、搜索、关闭等多个方法），违反接口隔离原则

重构后使用 Eino 标准的 SchemaDocumentWriter/Reader 接口：
- 与 Eino 生态完全兼容，可无缝接入其他 Eino 组件
- 业务层依赖窄接口，易于替换向量数据库实现
- 接口职责单一，符合接口隔离原则

#### mapper/repository 分层价值

**Mapper 层**：负责 schema.Document ↔ MilvusRow 的转换
- 隔离业务模型与存储模型
- 集中处理向量类型转换（float64 ↔ float32）
- 便于单元测试

**Repository 层**：负责 Milvus 数据库的直接操作
- 封装 Milvus SDK 细节
- 提供数据访问抽象
- 便于替换为其他向量数据库

**Store 层**：对外提供 SchemaDocumentWriter/Reader 接口
- 实现业务逻辑（Flush、LoadCollection 等）
- 协调 Mapper 和 Repository
- 保持接口简洁

#### schema.Document 的统一模型意义

- 业务层统一使用 Eino 标准的 schema.Document
- 消除了自定义 Document 结构带来的维护成本
- 与 Eino 生态组件（Indexer、Retriever）无缝集成
- 支持丰富的元数据（map[string]any）

#### 接口隔离 / 依赖倒置的应用

- **接口隔离**：SchemaDocumentWriter 只包含写入方法，SchemaDocumentReader 只包含读取方法
- **依赖倒置**：ingest/retrieval 服务依赖接口而非具体实现，main.go 负责注入具体实现
- **单一职责**：每个接口只负责一个明确的功能

### 🔄 数据流图

#### 文档入库流程

```
用户输入文本
    ↓
文本切分（TextSplitter）
    ↓
生成向量（Embedder）
    ↓
构建 schema.Document（业务层）
    ↓
Mapper: schema.Document → MilvusRow（转换层）
    ↓
Repository: 批量插入 Milvus（数据访问层）
    ↓
Flush（刷写到磁盘）
    ↓
LoadCollection（加载到内存）
```

#### 文档检索流程

```
用户查询
    ↓
生成查询向量（Embedder）
    ↓
向量类型转换（float64 → float32）
    ↓
Repository: Milvus 向量检索（数据访问层）
    ↓
Mapper: MilvusRow → schema.Document（转换层）
    ↓
Retriever: 返回 schema.Document（业务层）
    ↓
RAG: 结合检索结果生成答案
```

### 接口设计

- **SchemaDocumentWriter**: schema.Document 写入接口
  - `InsertSchemaDocuments(ctx, docs) ([]string, error)`
- **SchemaDocumentReader**: schema.Document 读取接口
  - `SearchSchemaDocuments(ctx, queryVec, topK) ([]*schema.Document, error)`

### 向量类型处理

- 业务层：使用 `[]float64`（豆包 embedding 返回类型）
- Milvus 内部：使用 `[]float32` 存储
- 转换层：通过 `utils.Float64ToFloat32` 和 `utils.Float32ToFloat64` 转换

### ⚙️ 技术选型说明

#### 为什么使用 Milvus

- **高性能**：基于 HNSW/IVF 索引，支持亿级向量毫秒级检索
- **开源生态**：Apache 2.0 协议，社区活跃，文档完善
- **云原生**：支持 Kubernetes 部署，易于扩展
- **功能丰富**：支持多种索引类型、距离度量、过滤查询

#### 为什么使用 Eino

- **标准接口**：提供统一的 AI 组件接口（Indexer、Retriever、Embedder）
- **组件化**：可灵活组合不同组件，易于扩展
- **生产级**：字节跳动内部生产环境验证，稳定性高
- **生态兼容**：与火山引擎、OpenAI 等主流服务无缝集成

#### embedding 模型选择

- **火山引擎 doubao-embedding-vision-251215**：
  - 中文优化，对中文语义理解更准确
  - 2048 维向量，兼顾精度和性能
  - 支持多模态（文本、图像）
  - API 稳定，成本低

### 🔌 可扩展性说明

#### 如何替换向量数据库

实现 SchemaDocumentWriter/Reader 接口即可：

```go
type MyVectorStore struct {
    // 实现细节
}

func (s *MyVectorStore) InsertSchemaDocuments(ctx context.Context, docs []*schema.Document) ([]string, error) {
    // 实现插入逻辑
}

func (s *MyVectorStore) SearchSchemaDocuments(ctx context.Context, queryVec []float32, topK int) ([]*schema.Document, error) {
    // 实现检索逻辑
}
```

然后在 main.go 中替换注入即可，无需修改 ingest/retrieval 服务代码。

#### 如何接入新的 embedding 模型

实现 Eino Embedder 接口：

```go
type MyEmbedder struct {
    // 实现细节
}

func (e *MyEmbedder) EmbedStrings(ctx context.Context, texts []string) ([][]float64, error) {
    // 调用 embedding 模型 API
}
```

然后在 main.go 中替换注入即可。

#### 如何扩展 Retriever

支持混合检索：

```go
// 在 retrieval/service.go 中扩展
func (s *Service) RetrieveHybrid(ctx context.Context, query string, opts ...retriever.Option) ([]*schema.Document, error) {
    // 1. 向量检索
    vecDocs, _ := s.Retrieve(ctx, query, opts...)

    // 2. 关键词检索（如 BM25）
    keywordDocs, _ := s.keywordSearch(query)

    // 3. 融合排序（如 RRF）
    return s.rerank(vecDocs, keywordDocs)
}
```

#### 如何扩展 RAG 策略

支持多轮对话和 context 管理：

```go
// 在 chat/chatbot.go 中扩展
type ChatBot struct {
    conversationHistory []Message
    contextManager     *ContextManager
}

func (c *ChatBot) Chat(ctx context.Context, query string) (string, error) {
    // 1. 从历史对话中提取上下文
    context := c.contextManager.ExtractContext(c.conversationHistory)

    // 2. 结合上下文进行检索
    docs, _ := c.retrievalService.RetrieveWithContext(ctx, query, context)

    // 3. 生成答案
    answer, _ := c.chatModel.Generate(ctx, query, docs)

    // 4. 更新对话历史
    c.conversationHistory = append(c.conversationHistory, Message{query, answer})

    return answer, nil
}
```

## 依赖说明

- **Eino**: v0.8.6 - 云原生 AI 组件框架
- **Milvus SDK**: v2 - Milvus 向量数据库客户端
- **火山引擎 SDK**: 豆包大模型和嵌入模型

## 更多信息

- [Eino 文档](https://github.com/cloudwego/eino)
- [Milvus 文档](https://milvus.io/docs)
- [MILVUS_INTEGRATION.md](./MILVUS_INTEGRATION.md) - Milvus 集成详细说明