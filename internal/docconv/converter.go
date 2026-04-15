package docconv

import (
	"eino-tutorial/internal/vectorstore"
	"github.com/cloudwego/eino/schema"
)

// SchemaToCustom 将 schema.Document 转换为自定义 Document
func SchemaToCustom(doc *schema.Document) (*vectorstore.Document, error) {
	customDoc := &vectorstore.Document{
		ID:       doc.ID,
		Content:  doc.Content,
		Vector:   doc.DenseVector(),
		Score:    doc.Score(),
		Metadata: make(map[string]string),
	}

	// 从 MetaData 中提取业务字段（doc_id, chunk_index, source 等）
	if doc.MetaData != nil {
		if val, ok := doc.MetaData["doc_id"].(string); ok {
			customDoc.DocID = val
		}
		if val, ok := doc.MetaData["chunk_index"].(int64); ok {
			customDoc.ChunkIndex = val
		}
		if val, ok := doc.MetaData["source"].(string); ok {
			customDoc.Source = val
		}

		// 复制其他字符串类型的元数据
		for k, v := range doc.MetaData {
			if str, ok := v.(string); ok {
				customDoc.Metadata[k] = str
			}
		}
	}

	return customDoc, nil
}

// CustomToSchema 将自定义 Document 转换为 schema.Document
func CustomToSchema(doc *vectorstore.Document) *schema.Document {
	metaData := make(map[string]any)

	// 存储业务字段到 MetaData
	if doc.DocID != "" {
		metaData["doc_id"] = doc.DocID
	}
	if doc.ChunkIndex != 0 {
		metaData["chunk_index"] = doc.ChunkIndex
	}
	if doc.Source != "" {
		metaData["source"] = doc.Source
	}

	// 复制元数据
	for k, v := range doc.Metadata {
		metaData[k] = v
	}

	schemaDoc := &schema.Document{
		ID:       doc.ID,
		Content:  doc.Content,
		MetaData: metaData,
	}

	// 使用标准方法设置向量和分数
	if doc.Vector != nil {
		schemaDoc = schemaDoc.WithDenseVector(doc.Vector)
	}
	if doc.Score != 0 {
		schemaDoc = schemaDoc.WithScore(doc.Score)
	}

	return schemaDoc
}
