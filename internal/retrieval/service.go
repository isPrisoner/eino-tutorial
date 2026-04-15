package retrieval

import (
	"context"
	"fmt"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/schema"

	"eino-tutorial/internal/docconv"
	"eino-tutorial/internal/vectorstore"
)

// Service 文档检索服务
type Service struct {
	ctx         context.Context
	embedder    embedding.Embedder
	vectorstore vectorstore.VectorStore
	// 注意：ctx 作为字段是阶段性设计，后续标准化时应改为由方法显式传入 ctx
}

// NewService 创建文档检索服务
func NewService(ctx context.Context, embedder embedding.Embedder, vs vectorstore.VectorStore) *Service {
	return &Service{
		ctx:         ctx,
		embedder:    embedder,
		vectorstore: vs,
	}
}

// SearchDocuments 搜索相关文档
func (s *Service) SearchDocuments(query string, topK int) ([]*vectorstore.Document, error) {
	// 桥接到 Retrieve 标准接口
	schemaDocs, err := s.Retrieve(s.ctx, query, retriever.WithTopK(topK))
	if err != nil {
		return nil, fmt.Errorf("检索文档失败: %w", err)
	}

	// 转换为 vectorstore.Document
	customDocs := make([]*vectorstore.Document, 0, len(schemaDocs))
	for _, doc := range schemaDocs {
		customDoc, err := docconv.SchemaToCustom(doc)
		if err != nil {
			return nil, fmt.Errorf("文档转换失败: %w", err)
		}
		customDocs = append(customDocs, customDoc)
	}

	return customDocs, nil
}

// Retrieve 实现 Eino Retriever 接口，检索文档
func (s *Service) Retrieve(ctx context.Context, query string, opts ...retriever.Option) ([]*schema.Document, error) {
	if s.vectorstore == nil {
		return nil, fmt.Errorf("向量存储未初始化")
	}

	// 使用 Eino 官方推荐的 GetCommonOptions 解析选项
	commonOpts := retriever.GetCommonOptions(&retriever.Options{}, opts...)

	// 如果传入的 Embedding 选项不为空，优先使用
	embedder := commonOpts.Embedding
	if embedder == nil {
		embedder = s.embedder
	}

	if embedder == nil {
		return nil, fmt.Errorf("嵌入器未初始化，请通过 retriever.WithEmbedding 选项传入")
	}

	// 使用 TopK 选项，默认值为 10
	topK := 10
	if commonOpts.TopK != nil {
		topK = *commonOpts.TopK
	}

	// ScoreThreshold 等其他选项通过 GetCommonOptions 正确接收，但本阶段暂不执行

	// 生成查询向量
	embeddings, err := embedder.EmbedStrings(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("生成查询向量失败: %w", err)
	}

	queryEmbedding := embeddings[0]

	// 使用 Milvus 搜索
	customDocs, err := s.vectorstore.Search(ctx, queryEmbedding, topK)
	if err != nil {
		return nil, fmt.Errorf("搜索失败: %w", err)
	}

	// 转换为 schema.Document
	schemaDocs := make([]*schema.Document, 0, len(customDocs))
	for _, doc := range customDocs {
		schemaDocs = append(schemaDocs, docconv.CustomToSchema(doc))
	}

	return schemaDocs, nil
}
