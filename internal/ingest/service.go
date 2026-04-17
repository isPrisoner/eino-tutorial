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
	"eino-tutorial/internal/transformer"
	"eino-tutorial/internal/utils"
	"eino-tutorial/internal/vectorstore"
)

// Service 文档入库服务
type Service struct {
	ctx          context.Context
	embedder     embedding.Embedder
	schemaWriter vectorstore.SchemaDocumentWriter
	textsplitter *textsplitter.TextSplitter
	transformer  *transformer.Router
	// ctx 作为字段存储，后续标准化时应改为由方法显式传入 ctx
}

// NewService 创建文档入库服务（使用旧 textsplitter，保持向后兼容）
func NewService(ctx context.Context, embedder embedding.Embedder, writer vectorstore.SchemaDocumentWriter, ts *textsplitter.TextSplitter) *Service {
	return &Service{
		ctx:          ctx,
		embedder:     embedder,
		schemaWriter: writer,
		textsplitter: ts,
	}
}

// NewServiceWithTransformer 创建文档入库服务（使用新 transformer）
func NewServiceWithTransformer(ctx context.Context, embedder embedding.Embedder, writer vectorstore.SchemaDocumentWriter, tr *transformer.Router) *Service {
	return &Service{
		ctx:          ctx,
		embedder:     embedder,
		schemaWriter: writer,
		transformer:  tr,
	}
}

// ImportText 导入文本内容（应用层导入入口）
// id: 文档 ID（可以为空，自动生成）
// content: 文档内容
// source: 来源标识（如 "manual"、"file" 等）
// 返回: 文档 ID 和分块数量
func (s *Service) ImportText(id, content, source string) (docID string, chunkCount int, err error) {
	// 优先使用 transformer，如果不可用则使用 textsplitter（向后兼容）
	if s.transformer != nil {
		return s.importTextWithTransformer(id, content, source)
	}

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

	// 3. 调用 Store 方法
	_, err = s.Store(s.ctx, schemaDocs)
	if err != nil {
		return "", 0, fmt.Errorf("存储文档失败: %w", err)
	}

	utils.DebugLog("成功导入文本 %s，切分为 %d 个分块", id, len(chunks))
	return id, len(chunks), nil
}

// importTextWithTransformer 使用 transformer 进行文本导入
func (s *Service) importTextWithTransformer(id, content, source string) (docID string, chunkCount int, err error) {
	// 如果 id 为空，自动生成
	if id == "" {
		id = uuid.New().String()
	}

	// 1. 构建单个 schema.Document（包含完整内容）
	metadata := map[string]any{
		"doc_id":          id,
		"source":          source,
		"original_length": strconv.Itoa(len(content)),
	}
	schemaDoc := docconv.BuildSchemaDocument(uuid.New().String(), content, metadata)

	// 2. 调用 transformer.Transform() 进行切分
	transformedDocs, err := s.transformer.Transform(s.ctx, []*schema.Document{schemaDoc})
	if err != nil {
		return "", 0, fmt.Errorf("文档切分失败: %w", err)
	}

	if len(transformedDocs) == 0 {
		return "", 0, fmt.Errorf("文档切分后没有有效分块")
	}

	// 3. 为切分后的文档补充 chunk_index metadata
	for i, doc := range transformedDocs {
		if doc.MetaData == nil {
			doc.MetaData = make(map[string]any)
		}
		doc.MetaData["chunk_index"] = int64(i)
		// 确保保留原始的 doc_id
		doc.MetaData["doc_id"] = id
	}

	// 4. 调用 Store 方法
	_, err = s.Store(s.ctx, transformedDocs)
	if err != nil {
		return "", 0, fmt.Errorf("存储文档失败: %w", err)
	}

	utils.DebugLog("成功导入文本 %s，切分为 %d 个分块", id, len(transformedDocs))
	return id, len(transformedDocs), nil
}

// ImportFile 导入文件（应用层导入入口）
// filePath: 文件路径
// 返回: 文档 ID 和分块数量
func (s *Service) ImportFile(filePath string) (docID string, chunkCount int, err error) {
	// 优先使用 transformer，如果不可用则使用 textsplitter（向后兼容）
	if s.transformer != nil {
		return s.importFileWithTransformer(filePath)
	}

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

	// 6. 调用 Store 方法
	_, err = s.Store(s.ctx, schemaDocs)
	if err != nil {
		return "", 0, fmt.Errorf("存储文档失败: %w", err)
	}

	utils.DebugLog("成功导入文件 %s，切分为 %d 个分块", docID, len(chunks))
	return docID, len(chunks), nil
}

// importFileWithTransformer 使用 transformer 进行文件导入
func (s *Service) importFileWithTransformer(filePath string) (docID string, chunkCount int, err error) {
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

	// 4. 构建单个 schema.Document（包含完整内容）
	metadata := map[string]any{
		"doc_id":          docID,
		"source":          absPath,
		"filename":        filepath.Base(filePath),
		"filepath":        absPath,
		"original_length": strconv.Itoa(len(content)),
		"file_type":       strings.TrimPrefix(ext, "."),
	}
	schemaDoc := docconv.BuildSchemaDocument(uuid.New().String(), content, metadata)

	// 5. 调用 transformer.Transform() 进行切分（Router 会根据文件扩展名选择合适的 splitter）
	transformedDocs, err := s.transformer.Transform(s.ctx, []*schema.Document{schemaDoc})
	if err != nil {
		return "", 0, fmt.Errorf("文档切分失败: %w", err)
	}

	if len(transformedDocs) == 0 {
		return "", 0, fmt.Errorf("文档切分后没有有效分块")
	}

	// 6. 为切分后的文档补充 chunk_index metadata
	for i, doc := range transformedDocs {
		if doc.MetaData == nil {
			doc.MetaData = make(map[string]any)
		}
		doc.MetaData["chunk_index"] = int64(i)
		// 确保保留原始的 doc_id 和其他文件相关 metadata
		doc.MetaData["doc_id"] = docID
		doc.MetaData["source"] = absPath
		doc.MetaData["filename"] = filepath.Base(filePath)
		doc.MetaData["filepath"] = absPath
		doc.MetaData["file_type"] = strings.TrimPrefix(ext, ".")
	}

	// 7. 调用 Store 方法
	_, err = s.Store(s.ctx, transformedDocs)
	if err != nil {
		return "", 0, fmt.Errorf("存储文档失败: %w", err)
	}

	utils.DebugLog("成功导入文件 %s，切分为 %d 个分块", docID, len(transformedDocs))
	return docID, len(transformedDocs), nil
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
