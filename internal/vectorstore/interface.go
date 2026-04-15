package vectorstore

import (
	"context"

	"github.com/cloudwego/eino/schema"
)

// SchemaDocumentWriter schema.Document 写入接口
type SchemaDocumentWriter interface {
	// InsertSchemaDocuments 批量插入 schema.Document
	InsertSchemaDocuments(ctx context.Context, docs []*schema.Document) ([]string, error)
}

// SchemaDocumentReader schema.Document 读取接口
type SchemaDocumentReader interface {
	// SearchSchemaDocuments 搜索返回 schema.Document
	SearchSchemaDocuments(ctx context.Context, queryVec []float32, topK int) ([]*schema.Document, error)
}
