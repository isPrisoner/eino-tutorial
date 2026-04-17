package transformer

import (
	"context"

	markdownsplitter "github.com/cloudwego/eino-ext/components/document/transformer/splitter/markdown"
	"github.com/cloudwego/eino/components/document"
	"github.com/cloudwego/eino/schema"
)

// MarkdownSplitter 封装 Eino 的 Markdown Header Splitter
// 用于 .md 和 .markdown 文件，按标题层级切分
type MarkdownSplitter struct {
	transformer document.Transformer
}

// NewMarkdownSplitter 创建 Markdown Splitter
func NewMarkdownSplitter(ctx context.Context) (*MarkdownSplitter, error) {
	// 使用 eino-ext 官方的 Markdown Header Splitter
	transformer, err := markdownsplitter.NewHeaderSplitter(ctx, &markdownsplitter.HeaderConfig{
		Headers: map[string]string{
			"##":  "",
			"###": "",
		},
	})
	if err != nil {
		return nil, err
	}

	return &MarkdownSplitter{
		transformer: transformer,
	}, nil
}

// Transform 实现 document.Transformer 接口
func (ms *MarkdownSplitter) Transform(ctx context.Context, src []*schema.Document, opts ...document.TransformerOption) ([]*schema.Document, error) {
	return ms.transformer.Transform(ctx, src, opts...)
}
