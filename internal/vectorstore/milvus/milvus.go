package milvus

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/google/uuid"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"eino-tutorial/internal/vectorstore"
)

var debugMode = os.Getenv("DEBUG") == "true"

// debugLog 仅在 DEBUG=true 时输出日志
func debugLog(format string, args ...interface{}) {
	if debugMode {
		log.Printf(format, args...)
	}
}

// MilvusStore Milvus 向量存储实现
type MilvusStore struct {
	client         client.Client
	address        string
	collectionName string
	dim            int
	topK           int
}

// NewMilvusStore 创建 Milvus 存储实例
func NewMilvusStore(ctx context.Context, address string, dim, topK int) (*MilvusStore, error) {
	cli, err := client.NewGrpcClient(ctx, address)
	if err != nil {
		return nil, fmt.Errorf("创建 Milvus 客户端失败: %w", err)
	}

	// 初始化 collection
	if err := initCollection(ctx, cli); err != nil {
		cli.Close()
		return nil, fmt.Errorf("初始化 collection 失败: %w", err)
	}

	return &MilvusStore{
		client:         cli,
		address:        address,
		collectionName: collectionName,
		dim:            dim,
		topK:           topK,
	}, nil
}

// InsertDocuments 批量插入文档向量
func (ms *MilvusStore) InsertDocuments(ctx context.Context, docs []*vectorstore.Document) error {
	if len(docs) == 0 {
		return nil
	}

	// 准备数据
	ids := make([]string, 0, len(docs))
	contents := make([]string, 0, len(docs))
	vectors := make([][]float32, 0, len(docs))
	docIDs := make([]string, 0, len(docs))
	chunkIndexes := make([]int64, 0, len(docs))
	sources := make([]string, 0, len(docs))
	metadataJSONs := make([]string, 0, len(docs))

	for _, doc := range docs {
		// 生成 ID（如果未提供）
		if doc.ID == "" {
			doc.ID = uuid.New().String()
		}

		// 校验向量维度
		if len(doc.Vector) != ms.dim {
			return fmt.Errorf("向量维度不匹配: 期望 %d, 实际 %d", ms.dim, len(doc.Vector))
		}

		ids = append(ids, doc.ID)
		contents = append(contents, doc.Content)
		docIDs = append(docIDs, doc.DocID)
		chunkIndexes = append(chunkIndexes, doc.ChunkIndex)
		sources = append(sources, doc.Source)

		// 转换向量 float64 → float32
		vec := make([]float32, len(doc.Vector))
		for i, v := range doc.Vector {
			vec[i] = float32(v)
		}
		vectors = append(vectors, vec)

		// 序列化元数据
		if doc.Metadata != nil {
			metadataJSON, err := json.Marshal(doc.Metadata)
			if err != nil {
				return fmt.Errorf("序列化元数据失败: %w", err)
			}
			metadataJSONs = append(metadataJSONs, string(metadataJSON))
		} else {
			metadataJSONs = append(metadataJSONs, "{}")
		}
	}

	// 插入数据
	_, err := ms.client.Insert(
		ctx,
		ms.collectionName,
		"",
		entity.NewColumnVarChar("id", ids),
		entity.NewColumnVarChar("content", contents),
		entity.NewColumnFloatVector("vector", ms.dim, vectors),
		entity.NewColumnVarChar("doc_id", docIDs),
		entity.NewColumnInt64("chunk_index", chunkIndexes),
		entity.NewColumnVarChar("source", sources),
		entity.NewColumnVarChar("metadata_json", metadataJSONs),
	)
	if err != nil {
		return fmt.Errorf("插入数据失败: %w", err)
	}

	debugLog("成功插入 %d 个文档向量", len(docs))

	// Flush 确保数据写入磁盘
	if err := ms.client.Flush(ctx, ms.collectionName, false); err != nil {
		return fmt.Errorf("Flush 失败: %w", err)
	}

	// LoadCollection 确保数据可搜索
	if err := ms.client.LoadCollection(ctx, ms.collectionName, false); err != nil {
		return fmt.Errorf("LoadCollection 失败: %w", err)
	}

	return nil
}

// Search 搜索相似文档
func (ms *MilvusStore) Search(ctx context.Context, queryEmbedding []float64, topK int) ([]*vectorstore.Document, error) {
	// 校验查询向量维度
	if len(queryEmbedding) != ms.dim {
		return nil, fmt.Errorf("查询向量维度不匹配: 期望 %d, 实际 %d", ms.dim, len(queryEmbedding))
	}

	// 转换查询向量 float64 → float32
	queryVec := make([]float32, len(queryEmbedding))
	for i, v := range queryEmbedding {
		queryVec[i] = float32(v)
	}

	// 执行搜索，指定 output_fields
	vectorField := "vector"
	outputFields := []string{"content", "doc_id", "chunk_index", "source", "metadata_json"}
	sp, _ := entity.NewIndexIvfFlatSearchParam(128)
	searchResult, err := ms.client.Search(
		ctx,
		ms.collectionName,
		[]string{},
		"",
		outputFields,
		[]entity.Vector{entity.FloatVector(queryVec)},
		vectorField,
		entity.COSINE,
		topK,
		sp,
	)
	if err != nil {
		return nil, fmt.Errorf("搜索失败: %w", err)
	}

	debugLog("搜索命中数: %d", len(searchResult))

	// 解析结果
	docs := make([]*vectorstore.Document, 0)
	for _, res := range searchResult {
		// 获取 content 列（必须字段）
		contentCol := res.Fields.GetColumn("content")
		if contentCol == nil {
			debugLog("警告: 未找到 content 列，跳过此结果")
			continue
		}
		contentVarCol, ok := contentCol.(*entity.ColumnVarChar)
		if !ok {
			debugLog("警告: content 列类型错误，跳过此结果")
			continue
		}
		contentData := contentVarCol.Data()
		if len(contentData) == 0 {
			debugLog("警告: content 列为空，跳过此结果")
			continue
		}

		// 获取 ID 列（可选，从 SearchResult.IDs 读取）
		var idData []string
		if res.IDs != nil {
			// 尝试从 IDs 读取主键
			if idVarCol, ok := res.IDs.(*entity.ColumnVarChar); ok {
				idData = idVarCol.Data()
			} else if idInt64Col, ok := res.IDs.(*entity.ColumnInt64); ok {
				// 如果是 Int64 类型的 ID
				idInt64Data := idInt64Col.Data()
				idData = make([]string, len(idInt64Data))
				for i, v := range idInt64Data {
					idData[i] = fmt.Sprintf("%d", v)
				}
			}
		}

		// 如果 IDs 也没有，尝试从 Fields 获取 id
		if len(idData) == 0 {
			idCol := res.Fields.GetColumn("id")
			if idCol != nil {
				if idVarCol, ok := idCol.(*entity.ColumnVarChar); ok {
					idData = idVarCol.Data()
				}
			}
		}

		if len(idData) == 0 {
			debugLog("警告: 未找到 id，将使用索引作为 id")
		}

		// 逐条解析每个 hit
		for i := 0; i < len(contentData); i++ {
			doc := &vectorstore.Document{
				Content: contentData[i],
				Score:   float64(res.Scores[i]),
			}

			// 设置 ID
			if len(idData) > i {
				doc.ID = idData[i]
			} else {
				doc.ID = fmt.Sprintf("hit_%d", i)
			}

			// 提取 doc_id（可选）
			if docIDCol := res.Fields.GetColumn("doc_id"); docIDCol != nil {
				if col, ok := docIDCol.(*entity.ColumnVarChar); ok {
					data := col.Data()
					if i < len(data) {
						doc.DocID = data[i]
					}
				}
			}

			// 提取 chunk_index（可选）
			if chunkIndexCol := res.Fields.GetColumn("chunk_index"); chunkIndexCol != nil {
				if col, ok := chunkIndexCol.(*entity.ColumnInt64); ok {
					data := col.Data()
					if i < len(data) {
						doc.ChunkIndex = data[i]
					}
				}
			}

			// 提取 source（可选）
			if sourceCol := res.Fields.GetColumn("source"); sourceCol != nil {
				if col, ok := sourceCol.(*entity.ColumnVarChar); ok {
					data := col.Data()
					if i < len(data) {
						doc.Source = data[i]
					}
				}
			}

			// 提取 metadata_json（可选）
			if metadataCol := res.Fields.GetColumn("metadata_json"); metadataCol != nil {
				if col, ok := metadataCol.(*entity.ColumnVarChar); ok {
					data := col.Data()
					if i < len(data) {
						var metadata map[string]string
						if err := json.Unmarshal([]byte(data[i]), &metadata); err == nil {
							doc.Metadata = metadata
						}
					}
				}
			}

			debugLog("命中[%d]: content=%s, score=%.4f, doc_id=%s, chunk_index=%d", i, doc.Content, doc.Score, doc.DocID, doc.ChunkIndex)
			docs = append(docs, doc)
		}
	}

	return docs, nil
}

// Close 关闭连接
func (ms *MilvusStore) Close() error {
	return ms.client.Close()
}
