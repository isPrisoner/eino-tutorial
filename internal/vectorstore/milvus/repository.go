package milvus

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// Repository 数据访问接口
type Repository interface {
	InsertRows(ctx context.Context, rows []*MilvusRow) error
	SearchRows(ctx context.Context, queryVec []float32, topK int) ([]*MilvusRow, error)
	Close() error
}

// MilvusRepository Milvus 数据访问实现
type MilvusRepository struct {
	client         client.Client
	collectionName string
	dim            int
}

// NewMilvusRepository 创建 Milvus Repository 实例
func NewMilvusRepository(cli client.Client, collectionName string, dim int) *MilvusRepository {
	return &MilvusRepository{
		client:         cli,
		collectionName: collectionName,
		dim:            dim,
	}
}

// InsertRows 批量插入 MilvusRow
func (mr *MilvusRepository) InsertRows(ctx context.Context, rows []*MilvusRow) error {
	if len(rows) == 0 {
		return nil
	}

	// 准备数据
	ids := make([]string, 0, len(rows))
	contents := make([]string, 0, len(rows))
	vectors := make([][]float32, 0, len(rows))
	docIDs := make([]string, 0, len(rows))
	chunkIndexes := make([]int64, 0, len(rows))
	sources := make([]string, 0, len(rows))
	metadataJSONs := make([]string, 0, len(rows))

	for _, row := range rows {
		// 校验向量维度
		if len(row.Vector) != mr.dim {
			return fmt.Errorf("向量维度不匹配: 期望 %d, 实际 %d", mr.dim, len(row.Vector))
		}

		ids = append(ids, row.ID)
		contents = append(contents, row.Content)
		docIDs = append(docIDs, row.DocID)
		chunkIndexes = append(chunkIndexes, row.ChunkIndex)
		sources = append(sources, row.Source)
		vectors = append(vectors, row.Vector)
		metadataJSONs = append(metadataJSONs, row.MetadataJSON)
	}

	// 插入数据
	_, err := mr.client.Insert(
		ctx,
		mr.collectionName,
		"",
		entity.NewColumnVarChar("id", ids),
		entity.NewColumnVarChar("content", contents),
		entity.NewColumnFloatVector("vector", mr.dim, vectors),
		entity.NewColumnVarChar("doc_id", docIDs),
		entity.NewColumnInt64("chunk_index", chunkIndexes),
		entity.NewColumnVarChar("source", sources),
		entity.NewColumnVarChar("metadata_json", metadataJSONs),
	)
	if err != nil {
		return fmt.Errorf("插入数据失败: %w", err)
	}

	debugLog("成功插入 %d 个文档向量", len(rows))

	// Flush 确保数据写入磁盘
	if err := mr.client.Flush(ctx, mr.collectionName, false); err != nil {
		return fmt.Errorf("Flush 失败: %w", err)
	}

	// LoadCollection 确保数据可搜索
	if err := mr.client.LoadCollection(ctx, mr.collectionName, false); err != nil {
		return fmt.Errorf("LoadCollection 失败: %w", err)
	}

	return nil
}

// SearchRows 搜索 MilvusRow
func (mr *MilvusRepository) SearchRows(ctx context.Context, queryVec []float32, topK int) ([]*MilvusRow, error) {
	// 校验查询向量维度
	if len(queryVec) != mr.dim {
		return nil, fmt.Errorf("查询向量维度不匹配: 期望 %d, 实际 %d", mr.dim, len(queryVec))
	}

	// 执行搜索，指定 output_fields
	vectorField := "vector"
	outputFields := []string{"content", "doc_id", "chunk_index", "source", "metadata_json"}
	sp, _ := entity.NewIndexIvfFlatSearchParam(128)
	searchResult, err := mr.client.Search(
		ctx,
		mr.collectionName,
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

	// 解析结果为 MilvusRow 列表
	rows := make([]*MilvusRow, 0)
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

		// 获取 ID 列
		var idData []string
		if res.IDs != nil {
			if idVarCol, ok := res.IDs.(*entity.ColumnVarChar); ok {
				idData = idVarCol.Data()
			} else if idInt64Col, ok := res.IDs.(*entity.ColumnInt64); ok {
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

		// 获取 doc_id 列
		var docIDData []string
		if docIDCol := res.Fields.GetColumn("doc_id"); docIDCol != nil {
			if col, ok := docIDCol.(*entity.ColumnVarChar); ok {
				docIDData = col.Data()
			}
		}

		// 获取 chunk_index 列
		var chunkIndexData []int64
		if chunkIndexCol := res.Fields.GetColumn("chunk_index"); chunkIndexCol != nil {
			if col, ok := chunkIndexCol.(*entity.ColumnInt64); ok {
				chunkIndexData = col.Data()
			}
		}

		// 获取 source 列
		var sourceData []string
		if sourceCol := res.Fields.GetColumn("source"); sourceCol != nil {
			if col, ok := sourceCol.(*entity.ColumnVarChar); ok {
				sourceData = col.Data()
			}
		}

		// 获取 metadata_json 列
		var metadataJSONData []string
		if metadataCol := res.Fields.GetColumn("metadata_json"); metadataCol != nil {
			if col, ok := metadataCol.(*entity.ColumnVarChar); ok {
				metadataJSONData = col.Data()
			}
		}

		// 逐条解析每个 hit
		for i := 0; i < len(contentData); i++ {
			row := &MilvusRow{
				Content: contentData[i],
				Score:   float64(res.Scores[i]),
			}

			// 设置 ID
			if len(idData) > i {
				row.ID = idData[i]
			} else {
				row.ID = fmt.Sprintf("hit_%d", i)
			}

			// 设置 doc_id
			if len(docIDData) > i {
				row.DocID = docIDData[i]
			}

			// 设置 chunk_index
			if len(chunkIndexData) > i {
				row.ChunkIndex = chunkIndexData[i]
			}

			// 设置 source
			if len(sourceData) > i {
				row.Source = sourceData[i]
			}

			// 设置 metadata_json
			if len(metadataJSONData) > i {
				row.MetadataJSON = metadataJSONData[i]
			}

			debugLog("命中[%d]: content=%s, score=%.4f, doc_id=%s, chunk_index=%d", i, row.Content, row.Score, row.DocID, row.ChunkIndex)
			rows = append(rows, row)
		}
	}

	return rows, nil
}

// Close 关闭连接
func (mr *MilvusRepository) Close() error {
	return mr.client.Close()
}
