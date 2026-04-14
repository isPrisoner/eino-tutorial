package retrieval

import (
	"context"
	"fmt"

	"github.com/cloudwego/eino/components/embedding"

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
	if s.embedder == nil {
		return nil, fmt.Errorf("嵌入器未初始化")
	}
	if s.vectorstore == nil {
		return nil, fmt.Errorf("向量存储未初始化")
	}

	// 生成查询向量
	embeddings, err := s.embedder.EmbedStrings(s.ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("生成查询向量失败: %w", err)
	}

	queryEmbedding := embeddings[0]

	// 使用 Milvus 搜索
	docs, err := s.vectorstore.Search(s.ctx, queryEmbedding, topK)
	if err != nil {
		return nil, fmt.Errorf("搜索失败: %w", err)
	}

	return docs, nil
}
