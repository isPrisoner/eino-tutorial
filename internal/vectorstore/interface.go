package vectorstore

import "context"

// Deprecated: Use github.com/cloudwego/eino/components/indexer.Indexer and retriever.Retriever instead.
// This interface will be removed in a future version.
type VectorStore interface {
	// InsertDocuments 批量插入文档向量
	InsertDocuments(ctx context.Context, docs []*Document) error

	// Search 搜索相似文档
	Search(ctx context.Context, queryEmbedding []float64, topK int) ([]*Document, error)

	// Close 关闭连接
	Close() error
}

// Deprecated: Use github.com/cloudwego/eino/schema.Document instead.
// This struct will be removed in a future version.
// Document 文档结构
type Document struct {
	ID         string            // 主键
	Content    string            // 文本内容
	Vector     []float64         // 向量（业务层使用 float64）
	DocID      string            // 文档 ID
	ChunkIndex int64             // 分块索引
	Source     string            // 来源
	Metadata   map[string]string // 元数据
	Score      float64           // 相似度分数（检索时）
}
