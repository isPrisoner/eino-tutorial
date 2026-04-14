package ingest

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/google/uuid"

	"eino-tutorial/internal/fileimport"
	"eino-tutorial/internal/textsplitter"
	"eino-tutorial/internal/utils"
	"eino-tutorial/internal/vectorstore"
)

// Service 文档入库服务
type Service struct {
	ctx          context.Context
	embedder     embedding.Embedder
	vectorstore  vectorstore.VectorStore
	textsplitter *textsplitter.TextSplitter
	// 注意：ctx 作为字段是阶段性设计，后续标准化时应改为由方法显式传入 ctx
}

// NewService 创建文档入库服务
func NewService(ctx context.Context, embedder embedding.Embedder, vs vectorstore.VectorStore, ts *textsplitter.TextSplitter) *Service {
	return &Service{
		ctx:          ctx,
		embedder:     embedder,
		vectorstore:  vs,
		textsplitter: ts,
	}
}

// AddDocument 添加文档到知识库
func (s *Service) AddDocument(id, content string) error {
	if s.embedder == nil {
		return fmt.Errorf("嵌入器未初始化")
	}
	if s.vectorstore == nil {
		return fmt.Errorf("向量存储未初始化")
	}
	if s.textsplitter == nil {
		return fmt.Errorf("文本切分器未初始化")
	}

	// 1. 文本切分
	chunks := s.textsplitter.Split(content)
	if len(chunks) == 0 {
		return fmt.Errorf("文本切分后没有有效分块")
	}

	// 2. 批量生成向量
	embeddings, err := s.embedder.EmbedStrings(s.ctx, chunks)
	if err != nil {
		return fmt.Errorf("生成文档向量失败: %w", err)
	}

	// 3. 构建文档列表
	docs := make([]*vectorstore.Document, 0, len(chunks))
	for i, chunk := range chunks {
		doc := &vectorstore.Document{
			ID:         uuid.New().String(),
			Content:    chunk,
			Vector:     embeddings[i],
			DocID:      id,
			ChunkIndex: int64(i),
			Source:     "manual",
			Metadata:   map[string]string{"original_length": strconv.Itoa(len(content))},
		}
		docs = append(docs, doc)
	}

	// 4. 批量插入 Milvus
	if err := s.vectorstore.InsertDocuments(s.ctx, docs); err != nil {
		return fmt.Errorf("插入向量存储失败: %w", err)
	}

	utils.DebugLog("成功添加文档 %s，切分为 %d 个分块", id, len(chunks))
	return nil
}

// AddFile 导入单个文件
func (s *Service) AddFile(filePath string) (*fileimport.FileImportResult, error) {
	result := &fileimport.FileImportResult{
		Success: false,
	}

	// 1. 检查文件是否存在
	if _, err := os.Stat(filePath); err != nil {
		result.Error = fmt.Errorf("文件不存在: %w", err)
		return result, result.Error
	}

	// 2. 检查文件扩展名
	ext := strings.ToLower(filepath.Ext(filePath))
	if ext != ".txt" && ext != ".md" {
		result.Error = fmt.Errorf("不支持的文件类型: %s（仅支持 .txt 和 .md）", ext)
		return result, result.Error
	}

	// 3. 读取文件内容
	content, err := fileimport.ReadFileContent(filePath)
	if err != nil {
		result.Error = fmt.Errorf("读取文件失败: %w", err)
		return result, result.Error
	}

	// 4. 获取相对路径作为 doc_id
	absPath, err := filepath.Abs(filePath)
	if err != nil {
		result.Error = fmt.Errorf("获取绝对路径失败: %w", err)
		return result, result.Error
	}

	docID, err := fileimport.GetRelativePath(absPath)
	if err != nil {
		result.Error = fmt.Errorf("计算相对路径失败: %w", err)
		return result, result.Error
	}

	// 5. 获取文件名
	filename := filepath.Base(filePath)

	// 6. 构建增强的元数据
	metadata := map[string]string{
		"filename":        filename,
		"filepath":        absPath,
		"original_length": strconv.Itoa(len(content)),
		"file_type":       strings.TrimPrefix(ext, "."),
	}

	// 7. 文本切分
	chunks := s.textsplitter.Split(content)
	if len(chunks) == 0 {
		result.Error = fmt.Errorf("文本切分后没有有效分块")
		return result, result.Error
	}

	// 8. 批量生成向量
	embeddings, err := s.embedder.EmbedStrings(s.ctx, chunks)
	if err != nil {
		result.Error = fmt.Errorf("生成文档向量失败: %w", err)
		return result, result.Error
	}

	// 9. 构建文档列表
	docs := make([]*vectorstore.Document, 0, len(chunks))
	for i, chunk := range chunks {
		doc := &vectorstore.Document{
			ID:         uuid.New().String(),
			Content:    chunk,
			Vector:     embeddings[i],
			DocID:      docID,
			ChunkIndex: int64(i),
			Source:     absPath,
			Metadata:   metadata,
		}
		docs = append(docs, doc)
	}

	// 10. 批量插入 Milvus
	if err := s.vectorstore.InsertDocuments(s.ctx, docs); err != nil {
		result.Error = fmt.Errorf("插入向量存储失败: %w", err)
		return result, result.Error
	}

	result.DocID = docID
	result.ChunkCount = len(chunks)
	result.Success = true

	utils.DebugLog("成功添加文件: %s, 切分为 %d 个分块", docID, len(chunks))
	return result, nil
}

// AddDir 批量导入目录中的文件
func (s *Service) AddDir(dirPath string) error {
	// 1. 检查目录是否存在
	if _, err := os.Stat(dirPath); err != nil {
		return fmt.Errorf("目录不存在: %w", err)
	}

	// 2. 扫描目录
	files, err := fileimport.ScanDirectory(dirPath)
	if err != nil {
		return fmt.Errorf("扫描目录失败: %w", err)
	}

	if len(files) == 0 {
		fmt.Printf("目录中没有找到 .txt 或 .md 文件\n")
		return nil
	}

	utils.DebugLog("找到 %d 个文件待导入", len(files))

	// 3. 批量导入
	successCount := 0
	failCount := 0
	totalChunks := 0
	var failedFiles []string

	for _, file := range files {
		utils.DebugLog("处理文件: %s", file)

		result, err := s.AddFile(file)
		if err != nil || !result.Success {
			utils.DebugLog("文件导入失败: %s, 错误: %v", file, err)
			failCount++
			failedFiles = append(failedFiles, file)
		} else {
			successCount++
			totalChunks += result.ChunkCount
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

	return nil
}
