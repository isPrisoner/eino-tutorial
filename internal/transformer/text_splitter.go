package transformer

import (
	"context"

	recursivesplitter "github.com/cloudwego/eino-ext/components/document/transformer/splitter/recursive"
	"github.com/cloudwego/eino/components/document"
	"github.com/cloudwego/eino/schema"
)

// TextSplitter 封装 Eino 的 Recursive/Text Splitter
// 用于 .txt 等非 Markdown 文件，保持原 CHUNK_SIZE/CHUNK_OVERLAP 语义
type TextSplitter struct {
	transformer document.Transformer
}

// NewTextSplitter 创建 Text Splitter
// chunkSize: 分块大小（字符数）
// chunkOverlap: 分块重叠（字符数）
func NewTextSplitter(chunkSize, chunkOverlap int) *TextSplitter {
	// 使用 eino-ext 官方的 Recursive Splitter
	transformer, _ := recursivesplitter.NewSplitter(context.Background(), &recursivesplitter.Config{
		ChunkSize:   chunkSize,
		OverlapSize: chunkOverlap,
	})

	return &TextSplitter{
		transformer: transformer,
	}
}

// Transform 实现 document.Transformer 接口
func (ts *TextSplitter) Transform(ctx context.Context, src []*schema.Document, opts ...document.TransformerOption) ([]*schema.Document, error) {
	return ts.transformer.Transform(ctx, src, opts...)
}
