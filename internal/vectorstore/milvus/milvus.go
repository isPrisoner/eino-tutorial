package milvus

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/milvus-io/milvus-sdk-go/v2/client"

	"eino-tutorial/internal/vectorstore"
)

var debugMode = os.Getenv("DEBUG") == "true"

// debugLog 仅在 DEBUG=true 时输出日志
func debugLog(format string, args ...interface{}) {
	if debugMode {
		log.Printf(format, args...)
	}
}

// MilvusStore Milvus 向量存储实现（过渡壳）
type MilvusStore struct {
	client         client.Client
	address        string
	collectionName string
	dim            int
	topK           int
	repository     Repository
}

// NewMilvusStore 创建 Milvus 存储实例
func NewMilvusStore(ctx context.Context, address string, dim, topK int) (*MilvusStore, error) {
	cli, err := client.NewGrpcClient(ctx, address)
	if err != nil {
		return nil, fmt.Errorf("创建 Milvus 客户端失败: %w", err)
	}

	// 初始化 collection
	if err := initCollection(ctx, cli); err != nil {
		cli.Close()
		return nil, fmt.Errorf("初始化 collection 失败: %w", err)
	}

	// 创建 repository
	repo := NewMilvusRepository(cli, collectionName, dim)

	return &MilvusStore{
		client:         cli,
		address:        address,
		collectionName: collectionName,
		dim:            dim,
		topK:           topK,
		repository:     repo,
	}, nil
}

// InsertDocuments 批量插入文档向量（桥接到 repository）
func (ms *MilvusStore) InsertDocuments(ctx context.Context, docs []*vectorstore.Document) error {
	if len(docs) == 0 {
		return nil
	}

	// 转换为 MilvusRow
	rows := make([]*MilvusRow, 0, len(docs))
	for _, doc := range docs {
		row, err := DocumentToRow(doc)
		if err != nil {
			return fmt.Errorf("文档转换失败: %w", err)
		}
		rows = append(rows, row)
	}

	// 调用 repository 插入
	if err := ms.repository.InsertRows(ctx, rows); err != nil {
		return fmt.Errorf("插入文档失败: %w", err)
	}

	return nil
}

// Search 搜索相似文档（桥接到 repository）
func (ms *MilvusStore) Search(ctx context.Context, queryEmbedding []float64, topK int) ([]*vectorstore.Document, error) {
	// 转换查询向量
	queryVec := float64ToFloat32(queryEmbedding)

	// 调用 repository 搜索
	rows, err := ms.repository.SearchRows(ctx, queryVec, topK)
	if err != nil {
		return nil, fmt.Errorf("搜索文档失败: %w", err)
	}

	// 转换为 vectorstore.Document
	docs := make([]*vectorstore.Document, 0, len(rows))
	for _, row := range rows {
		doc, err := RowToDocument(row)
		if err != nil {
			return nil, fmt.Errorf("文档转换失败: %w", err)
		}
		docs = append(docs, doc)
	}

	return docs, nil
}

// Close 关闭连接（桥接到 repository）
func (ms *MilvusStore) Close() error {
	return ms.repository.Close()
}
