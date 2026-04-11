package vectorstore

import "context"

// VectorStore 向量存储接口
type VectorStore interface {
	// InsertDocuments 批量插入文档向量
	InsertDocuments(ctx context.Context, docs []*Document) error

	// Search 搜索相似文档
	Search(ctx context.Context, queryEmbedding []float64, topK int) ([]*Document, error)

	// Close 关闭连接
	Close() error
}

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
