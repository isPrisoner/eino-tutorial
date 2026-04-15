# Milvus 向量存储集成说明

## 概述

本项目已成功集成 Milvus 向量数据库，使用 Eino 标准接口实现 RAG 功能。新的架构支持：

- Milvus 向量数据库存储和检索
- 基于 Eino schema.Document 的标准文档模型
- SchemaDocumentWriter/Reader 接口设计
- 文本自动切分
- 批量文档插入
- 向量维度自动校验
- IVF_FLAT 索引（COSINE 相似度）

## 环境变量配置

### 必需环境变量

```bash
# Ark API 配置
ARK_API_KEY=your_ark_api_key
ARK_MODEL_NAME=doubao-pro-32k
EMBEDDER=doubao-embedding-vision-251215
ARK_EMBEDDER_API_KEY=your_ark_api_key
```

### Milvus 配置

```bash
# Milvus 连接地址（默认：127.0.0.1:19530）
MILVUS_ADDRESS=127.0.0.1:19530

# 向量维度（默认：2048）
MILVUS_DIMENSION=2048

# 搜索返回数量（默认：3）
MILVUS_TOPK=3
```

### 文本切分配置

```bash
# 分块大小（默认：500）
CHUNK_SIZE=500

# 分块重叠（默认：50）
CHUNK_OVERLAP=50
```

## Milvus Collection Schema

```
Collection Name: eino_documents

Fields:
- id: VarChar(64) - 主键
- content: VarChar(16384) - 文本内容
- vector: FloatVector(2048) - 向量（float32）
- doc_id: VarChar(64) - 文档 ID
- chunk_index: Int64 - 分块索引
- source: VarChar(256) - 来源
- metadata_json: VarChar(4096) - 元数据 JSON

Index:
- Type: IVF_FLAT
- Metric: COSINE
- nlist: 128
```

## 使用说明

### 启动 Milvus

确保 Milvus 服务已启动：

```bash
# 使用 Docker 启动 Milvus
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest
```

### 运行程序

```bash
# 设置环境变量
export ARK_API_KEY=your_ark_api_key
export ARK_MODEL_NAME=doubao-pro-32k
export EMBEDDER=doubao-embedding-vision-251215
export ARK_EMBEDDER_API_KEY=your_ark_api_key

# 运行程序
go run cmd/main.go
```

### 添加文档

```
/add 这是要添加到知识库的文档内容
```

### 使用 RAG 查询

```
/rag 你的问题
```

## 架构说明

### 文件结构

```
internal/
├── vectorstore/
│   ├── interface.go      # SchemaDocumentWriter/Reader 接口
│   └── milvus/
│       ├── milvus.go    # MilvusStore 实现
│       ├── mapper.go    # schema.Document ↔ MilvusRow 转换
│       ├── repository.go # Milvus 数据访问层
│       └── collection.go # Collection 管理
└── textsplitter/
    └── textsplitter.go   # 文本切分实现
```

### 数据流程

1. **添加文档流程**：
   ```
   用户输入 → 文本切分 → 生成向量 → schema.Document → MilvusRow → Milvus 插入 → Flush → LoadCollection
   ```

2. **搜索流程**：
   ```
   用户查询 → 生成查询向量 → Milvus 搜索 → MilvusRow → schema.Document → RAG 生成
   ```

### 向量类型处理

- 业务层：使用 `[]float64`（豆包 embedding 返回类型）
- Milvus 内部：使用 `[]float32` 存储
- 转换层：通过 `utils.Float64ToFloat32` 和 `utils.Float32ToFloat64` 转换

### 分层设计

- **MilvusStore**: 对外提供 SchemaDocumentWriter/Reader 接口
- **Mapper**: 负责 schema.Document ↔ MilvusRow 的转换
- **Repository**: 负责 Milvus 数据库的直接操作
- **Collection**: 负责 Milvus Collection 的创建和管理

## 关键实现细节

### 0. VarChar 字段类型处理

**重要**：Milvus SDK 中 VarChar 字段必须使用对应的列构造器：

- **插入时**：使用 `entity.NewColumnVarChar` 而不是 `entity.NewColumnString`
- **查询时**：使用 `entity.ColumnVarChar` 而不是 `entity.ColumnString`

```go
// 正确：VarChar 字段插入
entity.NewColumnVarChar("id", ids)
entity.NewColumnVarChar("content", contents)
entity.NewColumnVarChar("doc_id", docIDs)
entity.NewColumnVarChar("source", sources)
entity.NewColumnVarChar("metadata_json", metadataJSONs)

// 正确：VarChar 字段查询解析
idVarCol, ok := idCol.(*entity.ColumnVarChar)
contentVarCol, ok := contentCol.(*entity.ColumnVarChar)
```

**错误原因**：
- `entity.NewColumnString` 创建的是 String 类型列，与 VarChar 字段定义不匹配
- Milvus 会报错：`param column id has type String but collection field definition is string`

### 1. InsertSchemaDocuments 后确保数据可搜索

```go
InsertSchemaDocuments 流程：
1. schema.Document 转换为 MilvusRow
2. Insert 批量插入
3. Flush 刷写到磁盘
4. LoadCollection 加载到内存
```

### 2. Search 指定 output_fields

```go
Search 时指定 output_fields：
- content
- doc_id
- chunk_index
- source
- metadata_json
```

### 3. metric_type 一致性

- 创建索引：`metric_type = entity.COSINE`
- 搜索查询：`metric_type = entity.COSINE`

### 4. 维度校验

- collection 不存在时，使用配置维度创建 schema
- collection 已存在时，读取 schema 并校验
- 写入和查询时都校验向量长度

### 5. 文本切分

- trim 空白
- 按字符数切分
- 过滤空 chunk
- 过滤过短 chunk（< 10 字符）
- 支持重叠

## 故障排查

### Milvus 连接失败

检查 Milvus 服务是否启动：

```bash
docker ps | grep milvus
```

### 向量维度不匹配

检查 `MILVUS_DIMENSION` 配置是否与 embedding 模型返回的维度一致。

### 搜索无结果

检查：
1. 是否成功插入数据
2. collection 是否已加载
3. 索引是否创建成功
