package transformer

import (
	"context"
	"path/filepath"
	"strings"

	"github.com/cloudwego/eino/components/document"
	"github.com/cloudwego/eino/schema"
)

// Router 根据文件类型选择对应的 splitter
// 职责边界：仅负责选择 splitter，不负责生成、补充或修改业务 metadata
type Router struct {
	markdownSplitter *MarkdownSplitter
	textSplitter     *TextSplitter
}

// NewRouter 创建 Router
func NewRouter(markdownSplitter *MarkdownSplitter, textSplitter *TextSplitter) *Router {
	return &Router{
		markdownSplitter: markdownSplitter,
		textSplitter:     textSplitter,
	}
}

// Transform 实现 document.Transformer 接口
// 根据 src 中的文件类型信息选择对应的 splitter
func (r *Router) Transform(ctx context.Context, src []*schema.Document, opts ...document.TransformerOption) ([]*schema.Document, error) {
	if len(src) == 0 {
		return src, nil
	}

	// 根据 src 中第一个文档的文件类型选择 splitter
	// 注意：这里假设 src 中的所有文档来自同一文件类型
	// 如果需要支持混合类型，需要逐个文档判断
	splitter := r.selectSplitter(src[0])

	return splitter.Transform(ctx, src, opts...)
}

// selectSplitter 根据文档选择对应的 splitter
// 路由规则：
// - .md/.markdown → Markdown Splitter
// - 其他文本文件 → Text Splitter
func (r *Router) selectSplitter(doc *schema.Document) document.Transformer {
	// 从 metadata 中获取文件路径
	source, ok := doc.MetaData["source"].(string)
	if !ok {
		// 如果没有 source 信息，默认使用 text splitter
		return r.textSplitter
	}

	ext := strings.ToLower(filepath.Ext(source))

	// .md 和 .markdown 文件使用 Markdown Splitter
	if ext == ".md" || ext == ".markdown" {
		return r.markdownSplitter
	}

	// 其他文件使用 Text Splitter
	return r.textSplitter
}

// TransformByFileType 根据文件类型显式选择 splitter 进行转换
// 这是一个辅助方法，用于已知文件类型的情况
func (r *Router) TransformByFileType(ctx context.Context, src []*schema.Document, fileType string, opts ...document.TransformerOption) ([]*schema.Document, error) {
	var splitter document.Transformer

	switch strings.ToLower(fileType) {
	case ".md", ".markdown":
		splitter = r.markdownSplitter
	default:
		splitter = r.textSplitter
	}

	return splitter.Transform(ctx, src, opts...)
}
