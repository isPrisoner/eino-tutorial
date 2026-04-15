package ingest

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/indexer"
	"github.com/cloudwego/eino/schema"
	"github.com/google/uuid"

	"eino-tutorial/internal/docconv"
	"eino-tutorial/internal/fileimport"
	"eino-tutorial/internal/textsplitter"
	"eino-tutorial/internal/utils"
	"eino-tutorial/internal/vectorstore"
)

// Service 文档入库服务
type Service struct {
	ctx          context.Context
	embedder     embedding.Embedder
	schemaWriter vectorstore.SchemaDocumentWriter
	textsplitter *textsplitter.TextSplitter
	// 注意：ctx 作为字段是阶段性设计，后续标准化时应改为由方法显式传入 ctx
}

// NewService 创建文档入库服务
func NewService(ctx context.Context, embedder embedding.Embedder, writer vectorstore.SchemaDocumentWriter, ts *textsplitter.TextSplitter) *Service {
	return &Service{
		ctx:          ctx,
		embedder:     embedder,
		schemaWriter: writer,
		textsplitter: ts,
	}
}

// ImportText 导入文本内容（应用层导入入口）
// id: 文档 ID（可以为空，自动生成）
// content: 文档内容
// source: 来源标识（如 "manual"、"file" 等）
// 返回: 文档 ID 和分块数量
func (s *Service) ImportText(id, content, source string) (docID string, chunkCount int, err error) {
	if s.textsplitter == nil {
		return "", 0, fmt.Errorf("文本切分器未初始化")
	}

	// 如果 id 为空，自动生成
	if id == "" {
		id = uuid.New().String()
	}

	// 1. 文本切分
	chunks := s.textsplitter.Split(content)
	if len(chunks) == 0 {
		return "", 0, fmt.Errorf("文本切分后没有有效分块")
	}

	// 2. 为每个 chunk 构建 schema.Document
	schemaDocs := make([]*schema.Document, 0, len(chunks))
	for i, chunk := range chunks {
		metadata := map[string]any{
			"doc_id":          id,
			"chunk_index":     int64(i),
			"source":          source,
			"original_length": strconv.Itoa(len(content)),
		}
		schemaDoc := docconv.BuildSchemaDocument(uuid.New().String(), chunk, metadata)
		schemaDocs = append(schemaDocs, schemaDoc)
	}

	// 3. 调用 Store 方法（桥接到标准接口）
	_, err = s.Store(s.ctx, schemaDocs)
	if err != nil {
		return "", 0, fmt.Errorf("存储文档失败: %w", err)
	}

	utils.DebugLog("成功导入文本 %s，切分为 %d 个分块", id, len(chunks))
	return id, len(chunks), nil
}

// ImportFile 导入文件（应用层导入入口）
// filePath: 文件路径
// 返回: 文档 ID 和分块数量
func (s *Service) ImportFile(filePath string) (docID string, chunkCount int, err error) {
	if s.textsplitter == nil {
		return "", 0, fmt.Errorf("文本切分器未初始化")
	}

	// 1. 读取文件内容
	content, err := fileimport.ReadFileContent(filePath)
	if err != nil {
		return "", 0, fmt.Errorf("读取文件失败: %w", err)
	}

	if len(content) == 0 {
		return "", 0, fmt.Errorf("文件内容为空")
	}

	// 2. 获取文件信息
	absPath, err := filepath.Abs(filePath)
	if err != nil {
		return "", 0, fmt.Errorf("获取文件绝对路径失败: %w", err)
	}

	ext := filepath.Ext(filePath)

	// 3. 生成文档 ID
	docID = uuid.New().String()

	// 4. 文本切分
	chunks := s.textsplitter.Split(content)
	if len(chunks) == 0 {
		return "", 0, fmt.Errorf("文本切分后没有有效分块")
	}

	// 5. 为每个 chunk 构建 schema.Document
	schemaDocs := make([]*schema.Document, 0, len(chunks))
	for i, chunk := range chunks {
		metadata := map[string]any{
			"doc_id":          docID,
			"chunk_index":     int64(i),
			"source":          absPath,
			"filename":        filepath.Base(filePath),
			"filepath":        absPath,
			"original_length": strconv.Itoa(len(content)),
			"file_type":       strings.TrimPrefix(ext, "."),
		}
		schemaDoc := docconv.BuildSchemaDocument(uuid.New().String(), chunk, metadata)
		schemaDocs = append(schemaDocs, schemaDoc)
	}

	// 6. 调用 Store 方法（桥接到标准接口）
	_, err = s.Store(s.ctx, schemaDocs)
	if err != nil {
		return "", 0, fmt.Errorf("存储文档失败: %w", err)
	}

	utils.DebugLog("成功导入文件 %s，切分为 %d 个分块", docID, len(chunks))
	return docID, len(chunks), nil
}

// ImportDir 导入目录（应用层导入入口）
// dirPath: 目录路径
// 返回: 成功导入的文件数量和总分块数量
func (s *Service) ImportDir(dirPath string) (successCount int, totalChunks int, err error) {
	// 1. 检查目录是否存在
	if _, err := os.Stat(dirPath); err != nil {
		return 0, 0, fmt.Errorf("目录不存在: %w", err)
	}

	// 2. 扫描目录
	files, err := fileimport.ScanDirectory(dirPath)
	if err != nil {
		return 0, 0, fmt.Errorf("扫描目录失败: %w", err)
	}

	if len(files) == 0 {
		fmt.Printf("目录中没有找到 .txt 或 .md 文件\n")
		return 0, 0, nil
	}

	utils.DebugLog("找到 %d 个文件待导入", len(files))

	// 3. 批量导入
	failCount := 0
	var failedFiles []string

	for _, file := range files {
		utils.DebugLog("处理文件: %s", file)

		_, chunkCount, err := s.ImportFile(file)
		if err != nil {
			utils.DebugLog("文件导入失败: %s, 错误: %v", file, err)
			failCount++
			failedFiles = append(failedFiles, file)
		} else {
			successCount++
			totalChunks += chunkCount
		}
	}

	// 4. 输出结果
	fmt.Printf("成功导入 %d 个文件，共 %d 个分块\n", successCount, totalChunks)
	if failCount > 0 {
		fmt.Printf("失败 %d 个文件\n", failCount)
		if utils.DebugMode {
			for _, file := range failedFiles {
				fmt.Printf("  - %s\n", file)
			}
		}
	}

	return successCount, totalChunks, nil
}

// Store 实现 Eino Indexer 接口，存储文档
func (s *Service) Store(ctx context.Context, docs []*schema.Document, opts ...indexer.Option) (ids []string, err error) {
	if s.schemaWriter == nil {
		return nil, fmt.Errorf("向量存储未初始化")
	}

	// 使用 Eino 官方推荐的 GetCommonOptions 解析选项
	commonOpts := indexer.GetCommonOptions(&indexer.Options{}, opts...)

	// 如果传入的 Embedding 选项不为空，优先使用
	embedder := commonOpts.Embedding
	if embedder == nil {
		embedder = s.embedder
	}

	if embedder == nil {
		return nil, fmt.Errorf("嵌入器未初始化，请通过 indexer.WithEmbedding 选项传入")
	}

	// 如果文档没有向量，则生成
	for i, doc := range docs {
		if doc.DenseVector() == nil || len(doc.DenseVector()) == 0 {
			embeddings, err := embedder.EmbedStrings(ctx, []string{doc.Content})
			if err != nil {
				return nil, fmt.Errorf("生成向量失败: %w", err)
			}
			// WithDenseVector 可能返回新对象，确保写回切片
			docs[i] = doc.WithDenseVector(embeddings[0])
		}
	}

	// 直接调用 InsertSchemaDocuments
	ids, err = s.schemaWriter.InsertSchemaDocuments(ctx, docs)
	if err != nil {
		return nil, fmt.Errorf("插入向量存储失败: %w", err)
	}

	return ids, nil
}
