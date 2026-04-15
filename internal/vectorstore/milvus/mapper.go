package milvus

import (
	"encoding/json"

	"github.com/cloudwego/eino/schema"

	"eino-tutorial/internal/utils"
)

// MilvusRow Milvus 持久化行模型
type MilvusRow struct {
	ID           string
	Content      string
	Vector       []float32
	DocID        string
	ChunkIndex   int64
	Source       string
	MetadataJSON string
	Score        float64 // 检索时的相似度分数
}

// SchemaToRow 将 schema.Document 转换为 MilvusRow
func SchemaToRow(doc *schema.Document) (*MilvusRow, error) {
	metadataJSON, err := serializeMetadataFromAny(doc.MetaData)
	if err != nil {
		return nil, err
	}

	vector := doc.DenseVector()
	var vector32 []float32
	if vector != nil {
		vector32 = utils.Float64ToFloat32(vector)
	}

	return &MilvusRow{
		ID:           doc.ID,
		Content:      doc.Content,
		Vector:       vector32,
		DocID:        getDocIDFromMetadata(doc.MetaData),
		ChunkIndex:   getChunkIndexFromMetadata(doc.MetaData),
		Source:       getSourceFromMetadata(doc.MetaData),
		MetadataJSON: metadataJSON,
		Score:        doc.Score(),
	}, nil
}

// RowToSchema 将 MilvusRow 转换为 schema.Document
func RowToSchema(row *MilvusRow) (*schema.Document, error) {
	metadata, err := deserializeMetadataToAny(row.MetadataJSON)
	if err != nil {
		return nil, err
	}

	// 确保业务字段在 metadata 中
	if row.DocID != "" {
		metadata["doc_id"] = row.DocID
	}
	if row.ChunkIndex != 0 {
		metadata["chunk_index"] = row.ChunkIndex
	}
	if row.Source != "" {
		metadata["source"] = row.Source
	}

	schemaDoc := &schema.Document{
		ID:       row.ID,
		Content:  row.Content,
		MetaData: metadata,
	}

	// 设置向量
	if row.Vector != nil {
		schemaDoc = schemaDoc.WithDenseVector(utils.Float32ToFloat64(row.Vector))
	}

	// 设置分数
	if row.Score != 0 {
		schemaDoc = schemaDoc.WithScore(row.Score)
	}

	return schemaDoc, nil
}

// getDocIDFromMetadata 从 metadata 中获取 doc_id
func getDocIDFromMetadata(metadata map[string]any) string {
	if metadata == nil {
		return ""
	}
	if val, ok := metadata["doc_id"].(string); ok {
		return val
	}
	return ""
}

// getChunkIndexFromMetadata 从 metadata 中获取 chunk_index
func getChunkIndexFromMetadata(metadata map[string]any) int64 {
	if metadata == nil {
		return 0
	}
	if val, ok := metadata["chunk_index"].(int64); ok {
		return val
	}
	if val, ok := metadata["chunk_index"].(int); ok {
		return int64(val)
	}
	if val, ok := metadata["chunk_index"].(float64); ok {
		return int64(val)
	}
	if val, ok := metadata["chunk_index"].(float32); ok {
		return int64(val)
	}
	return 0
}

// getSourceFromMetadata 从 metadata 中获取 source
func getSourceFromMetadata(metadata map[string]any) string {
	if metadata == nil {
		return ""
	}
	if val, ok := metadata["source"].(string); ok {
		return val
	}
	return ""
}

// serializeMetadataFromAny 将 map[string]any 序列化为 JSON 字符串
func serializeMetadataFromAny(metadata map[string]any) (string, error) {
	if metadata == nil {
		return "{}", nil
	}
	data, err := json.Marshal(metadata)
	if err != nil {
		return "{}", err
	}
	return string(data), nil
}

// deserializeMetadataToAny 将 JSON 字符串反序列化为 map[string]any
func deserializeMetadataToAny(metadataJSON string) (map[string]any, error) {
	if metadataJSON == "" {
		return make(map[string]any), nil
	}
	var metadata map[string]any
	err := json.Unmarshal([]byte(metadataJSON), &metadata)
	if err != nil {
		return make(map[string]any), nil
	}
	return metadata, nil
}
