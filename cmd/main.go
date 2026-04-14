package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	arkModel "github.com/cloudwego/eino-ext/components/model/ark"
	"github.com/cloudwego/eino/components/embedding"

	"eino-tutorial/internal/chat"
	"eino-tutorial/internal/cli"
	"eino-tutorial/internal/ingest"
	"eino-tutorial/internal/retrieval"
	"eino-tutorial/internal/textsplitter"
	"eino-tutorial/internal/utils"
	"eino-tutorial/internal/vectorstore"
	milvusStore "eino-tutorial/internal/vectorstore/milvus"
)

// getEnvFloat 从环境变量读取浮点数，提供默认值
func getEnvFloat(key string, defaultValue float64) float64 {
	if value := os.Getenv(key); value != "" {
		if f, err := strconv.ParseFloat(value, 64); err == nil {
			return f
		}
	}
	return defaultValue
}

// getEnvInt 从环境变量读取整数，提供默认值
func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if i, err := strconv.Atoi(value); err == nil {
			return i
		}
	}
	return defaultValue
}

func main() {
	ctx := context.Background()

	// 1. 创建 ChatModel (使用豆包 Ark)
	timeout := 30 * time.Second
	chatModel, err := arkModel.NewChatModel(ctx, &arkModel.ChatModelConfig{
		APIKey:  os.Getenv("ARK_API_KEY"),
		Model:   os.Getenv("ARK_MODEL_NAME"),
		Timeout: &timeout,
	})
	if err != nil {
		log.Fatalf("创建 ChatModel 失败: %v", err)
	}

	// 2. 创建嵌入器 (用于 RAG)
	var embedder embedding.Embedder
	if embedderModel := os.Getenv("EMBEDDER"); embedderModel != "" {
		// 使用火山引擎的嵌入器
		apiKey := os.Getenv("ARK_EMBEDDER_API_KEY")
		if apiKey == "" {
			log.Printf("未设置 ARK_EMBEDDER_API_KEY 环境变量，RAG功能将不可用")
			embedder = nil
		} else {
			embedder = chat.NewVolcengineEmbedder(apiKey, embedderModel)
			fmt.Println("RAG 功能已启用（使用火山引擎嵌入器）")
		}
	} else {
		fmt.Println("未设置 EMBEDDER 环境变量，RAG 功能将不可用")
	}

	// 3. 创建 Milvus 向量存储
	var vs vectorstore.VectorStore
	if embedder != nil {
		milvusAddress := os.Getenv("MILVUS_ADDRESS")
		if milvusAddress == "" {
			milvusAddress = "127.0.0.1:19530"
		}

		milvusDim := 2048
		if dimStr := os.Getenv("MILVUS_DIMENSION"); dimStr != "" {
			if d, err := strconv.Atoi(dimStr); err == nil {
				milvusDim = d
			}
		}

		milvusTopK := 3
		if topKStr := os.Getenv("MILVUS_TOPK"); topKStr != "" {
			if k, err := strconv.Atoi(topKStr); err == nil {
				milvusTopK = k
			}
		}

		store, err := milvusStore.NewMilvusStore(ctx, milvusAddress, milvusDim, milvusTopK)
		if err != nil {
			log.Printf("创建 Milvus 存储失败: %v (RAG功能将不可用)", err)
			vs = nil
		} else {
			vs = store
			utils.DebugLog("Milvus 向量存储已启用 (address=%s, dim=%d, topK=%d)", milvusAddress, milvusDim, milvusTopK)
		}
	} else {
		vs = nil
	}

	// 4. 创建文本切分器
	var splitter *textsplitter.TextSplitter
	if vs != nil {
		chunkSize := 500
		if sizeStr := os.Getenv("CHUNK_SIZE"); sizeStr != "" {
			if s, err := strconv.Atoi(sizeStr); err == nil {
				chunkSize = s
			}
		}

		chunkOverlap := 50
		if overlapStr := os.Getenv("CHUNK_OVERLAP"); overlapStr != "" {
			if o, err := strconv.Atoi(overlapStr); err == nil {
				chunkOverlap = o
			}
		}

		splitter = textsplitter.NewTextSplitter(chunkSize, chunkOverlap)
		utils.DebugLog("文本切分器已启用 (chunk_size=%d, chunk_overlap=%d)", chunkSize, chunkOverlap)
	} else {
		splitter = nil
	}

	// 5. 创建 ingest 服务
	var ingestService *ingest.Service
	if vs != nil && splitter != nil {
		ingestService = ingest.NewService(ctx, embedder, vs, splitter)
	}

	// 5.5 创建 retrieval 服务
	var retrievalService *retrieval.Service
	if embedder != nil && vs != nil {
		retrievalService = retrieval.NewService(ctx, embedder, vs)
	}

	// 6. 读取 RAG 配置
	ragMinScore := getEnvFloat("RAG_MIN_SCORE", 0.5)
	ragTopK := getEnvInt("RAG_TOPK", 5)
	ragMaxContextLen := getEnvInt("RAG_MAX_CONTEXT_LEN", 2000)
	ragMaxContextChunks := getEnvInt("RAG_MAX_CONTEXT_CHUNKS", 10)

	// 7. 创建 ChatBot
	chatBot := chat.NewChatBot(ctx, chatModel, retrievalService, ragMinScore, ragTopK, ragMaxContextLen, ragMaxContextChunks)

	// 8. 创建 CLI 处理器
	handler := cli.NewHandler(chatBot, ingestService)

	// 9. 启动 CLI
	if err := handler.Run(); err != nil {
		log.Fatalf("CLI 运行失败: %v", err)
	}
}
