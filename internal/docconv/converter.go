package docconv

import (
	"github.com/cloudwego/eino/schema"
)

// BuildSchemaDocument 构建一个 schema.Document，业务字段通过 metadata 统一传入
func BuildSchemaDocument(id, content string, metadata map[string]any) *schema.Document {
	if metadata == nil {
		metadata = make(map[string]any)
	}

	schemaDoc := &schema.Document{
		ID:       id,
		Content:  content,
		MetaData: metadata,
	}

	return schemaDoc
}
