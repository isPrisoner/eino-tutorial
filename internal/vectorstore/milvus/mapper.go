package milvus

import (
	"encoding/json"

	"eino-tutorial/internal/vectorstore"
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

// DocumentToRow 将 vectorstore.Document 转换为 MilvusRow
func DocumentToRow(doc *vectorstore.Document) (*MilvusRow, error) {
	metadataJSON, err := serializeMetadata(doc.Metadata)
	if err != nil {
		return nil, err
	}

	return &MilvusRow{
		ID:           doc.ID,
		Content:      doc.Content,
		Vector:       float64ToFloat32(doc.Vector),
		DocID:        doc.DocID,
		ChunkIndex:   doc.ChunkIndex,
		Source:       doc.Source,
		MetadataJSON: metadataJSON,
		Score:        doc.Score,
	}, nil
}

// RowToDocument 将 MilvusRow 转换为 vectorstore.Document
func RowToDocument(row *MilvusRow) (*vectorstore.Document, error) {
	metadata, err := deserializeMetadata(row.MetadataJSON)
	if err != nil {
		return nil, err
	}

	return &vectorstore.Document{
		ID:         row.ID,
		Content:    row.Content,
		Vector:     float32ToFloat64(row.Vector),
		DocID:      row.DocID,
		ChunkIndex: row.ChunkIndex,
		Source:     row.Source,
		Metadata:   metadata,
		Score:      row.Score,
	}, nil
}

// float64ToFloat32 将 float64 切片转换为 float32 切片
func float64ToFloat32(vec []float64) []float32 {
	if vec == nil {
		return nil
	}
	result := make([]float32, len(vec))
	for i, v := range vec {
		result[i] = float32(v)
	}
	return result
}

// float32ToFloat64 将 float32 切片转换为 float64 切片
func float32ToFloat64(vec []float32) []float64 {
	if vec == nil {
		return nil
	}
	result := make([]float64, len(vec))
	for i, v := range vec {
		result[i] = float64(v)
	}
	return result
}

// serializeMetadata 将元数据序列化为 JSON 字符串
func serializeMetadata(metadata map[string]string) (string, error) {
	if metadata == nil {
		return "{}", nil
	}
	data, err := json.Marshal(metadata)
	if err != nil {
		return "{}", err
	}
	return string(data), nil
}

// deserializeMetadata 将 JSON 字符串反序列化为元数据
func deserializeMetadata(metadataJSON string) (map[string]string, error) {
	if metadataJSON == "" {
		return make(map[string]string), nil
	}
	var metadata map[string]string
	err := json.Unmarshal([]byte(metadataJSON), &metadata)
	if err != nil {
		return make(map[string]string), nil
	}
	return metadata, nil
}
