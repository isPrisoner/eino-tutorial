package milvus

import (
	"context"
	"fmt"
	"log"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	collectionName = "eino_documents"
	dimension      = 2048 // 默认维度，会从实际 embedding 校验

	// 字段最大长度
	maxIDLength       = 64
	maxContentLength  = 16384
	maxDocIDLength    = 64
	maxSourceLength   = 256
	maxMetadataLength = 4096
)

// createSchema 创建 Collection Schema
func createSchema() *entity.Schema {
	return &entity.Schema{
		CollectionName: collectionName,
		Fields: []*entity.Field{
			{
				Name:       "id",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": fmt.Sprintf("%d", maxIDLength)},
				PrimaryKey: true,
				AutoID:     false,
			},
			{
				Name:       "content",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": fmt.Sprintf("%d", maxContentLength)},
			},
			{
				Name:       "vector",
				DataType:   entity.FieldTypeFloatVector,
				TypeParams: map[string]string{"dim": fmt.Sprintf("%d", dimension)},
			},
			{
				Name:       "doc_id",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": fmt.Sprintf("%d", maxDocIDLength)},
			},
			{
				Name:     "chunk_index",
				DataType: entity.FieldTypeInt64,
			},
			{
				Name:       "source",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": fmt.Sprintf("%d", maxSourceLength)},
			},
			{
				Name:       "metadata_json",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": fmt.Sprintf("%d", maxMetadataLength)},
			},
		},
	}
}

// createIndex 创建索引
func createIndex(ctx context.Context, cli client.Client, collectionName string) error {
	// 使用 IVF_FLAT 索引
	idx, err := entity.NewIndexIvfFlat(entity.COSINE, 128)
	if err != nil {
		return fmt.Errorf("创建 IVF_FLAT 索引失败: %w", err)
	}

	return cli.CreateIndex(ctx, collectionName, "vector", idx, false)
}

// initCollection 初始化 Collection
func initCollection(ctx context.Context, cli client.Client) error {
	// 检查 collection 是否存在
	has, err := cli.HasCollection(ctx, collectionName)
	if err != nil {
		return fmt.Errorf("检查 collection 失败: %w", err)
	}

	if has {
		log.Printf("Collection %s 已存在，跳过创建", collectionName)
		// TODO: 校验现有 schema 的维度
		return nil
	}

	// 创建 collection
	schema := createSchema()
	if err := cli.CreateCollection(ctx, schema, 0); err != nil {
		return fmt.Errorf("创建 collection 失败: %w", err)
	}

	log.Printf("Collection %s 创建成功", collectionName)

	// 创建索引
	if err := createIndex(ctx, cli, collectionName); err != nil {
		return fmt.Errorf("创建索引失败: %w", err)
	}

	log.Printf("索引创建成功")

	// 加载 collection
	if err := cli.LoadCollection(ctx, collectionName, false); err != nil {
		return fmt.Errorf("加载 collection 失败: %w", err)
	}

	log.Printf("Collection %s 加载成功", collectionName)

	return nil
}
