package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	arkModel "github.com/cloudwego/eino-ext/components/model/ark"
	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/schema"
	"github.com/google/uuid"
	"github.com/volcengine/volcengine-go-sdk/service/arkruntime"
	arkruntimeModel "github.com/volcengine/volcengine-go-sdk/service/arkruntime/model"

	"eino-tutorial/internal/fileimport"
	"eino-tutorial/internal/textsplitter"
	"eino-tutorial/internal/vectorstore"
	milvusStore "eino-tutorial/internal/vectorstore/milvus"
)

var (
	debugMode = os.Getenv("DEBUG") == "true"

	// RAG 配置（从环境变量读取，提供默认值）
	ragMinScore         = getEnvFloat("RAG_MIN_SCORE", 0.5)
	ragTopK             = getEnvInt("RAG_TOPK", 5)
	ragMaxContextLen    = getEnvInt("RAG_MAX_CONTEXT_LEN", 2000)
	ragMaxContextChunks = getEnvInt("RAG_MAX_CONTEXT_CHUNKS", 10)
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

// debugLog 仅在 DEBUG=true 时输出日志
func debugLog(format string, args ...interface{}) {
	if debugMode {
		log.Printf(format, args...)
	}
}

// ChatBot 结构体，使用 eino BaseChatModel 接口和 ChatTemplate
type ChatBot struct {
	model        model.BaseChatModel
	ctx          context.Context
	messages     []*schema.Message
	templates    map[string]prompt.ChatTemplate
	embedder     embedding.Embedder         // 嵌入器用于 RAG
	vectorstore  vectorstore.VectorStore    // 向量存储
	textsplitter *textsplitter.TextSplitter // 文本切分器
}

// VolcengineEmbedder 火山引擎嵌入器实现
type VolcengineEmbedder struct {
	client *arkruntime.Client
	model  string
}

// NewVolcengineEmbedder 创建火山引擎嵌入器
func NewVolcengineEmbedder(apiKey, model string) *VolcengineEmbedder {
	client := arkruntime.NewClientWithApiKey(apiKey)
	return &VolcengineEmbedder{
		client: client,
		model:  model,
	}
}

// EmbedStrings 实现 embedding.Embedder 接口
func (ve *VolcengineEmbedder) EmbedStrings(ctx context.Context, texts []string, opts ...embedding.Option) ([][]float64, error) {
	// 构建多模态嵌入请求
	inputs := make([]arkruntimeModel.MultimodalEmbeddingInput, len(texts))
	for i, text := range texts {
		inputs[i] = arkruntimeModel.MultimodalEmbeddingInput{
			Type: arkruntimeModel.MultiModalEmbeddingInputTypeText,
			Text: &text,
		}
	}

	req := arkruntimeModel.MultiModalEmbeddingRequest{
		Model: ve.model,
		Input: inputs,
	}

	// 调用多模态嵌入 API
	resp, err := ve.client.CreateMultiModalEmbeddings(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("调用火山引擎多模态嵌入 API 失败: %w", err)
	}

	// 提取向量 - 多模态响应的 Data.Embedding 直接是向量
	result := make([][]float64, len(texts))
	if len(resp.Data.Embedding) > 0 {
		// 将 float32 转换为 float64
		embedder := make([]float64, len(resp.Data.Embedding))
		for j, val := range resp.Data.Embedding {
			embedder[j] = float64(val)
		}
		result[0] = embedder
	}

	return result, nil
}

// NewChatBot 创建新的聊天机器人实例
func NewChatBot(ctx context.Context, chatModel model.BaseChatModel, embedder embedding.Embedder, vs vectorstore.VectorStore, ts *textsplitter.TextSplitter) *ChatBot {
	bot := &ChatBot{
		model: chatModel,
		ctx:   ctx,
		messages: []*schema.Message{
			schema.SystemMessage("你是一个友好的 AI 助手"),
		},
		templates:    make(map[string]prompt.ChatTemplate),
		embedder:     embedder,
		vectorstore:  vs,
		textsplitter: ts,
	}

	// 初始化模板
	bot.initTemplates()

	return bot
}

// initTemplates 初始化聊天模板
func (cb *ChatBot) initTemplates() {
	// 翻译模板
	cb.templates["translate"] = prompt.FromMessages(
		schema.FString,
		schema.SystemMessage("你是一个专业的翻译助手。请将用户输入的文本翻译成{target_lang}，只返回翻译结果，不要添加任何解释。"),
		schema.UserMessage("用户输入：{text}"),
	)

	// 代码生成模板
	cb.templates["code"] = prompt.FromMessages(
		schema.FString,
		schema.SystemMessage("你是一个专业的程序员。请根据用户的需求生成{language}代码，只返回代码，不要添加解释。"),
		schema.UserMessage("需求：{requirement}"),
	)

	// 总结模板
	cb.templates["summarize"] = prompt.FromMessages(
		schema.FString,
		schema.SystemMessage("你是一个专业的内容总结助手。请将用户提供的内容总结成{style}风格，控制在{max_length}字以内。"),
		schema.UserMessage("内容：{content}"),
	)
}

// UseTemplate 使用指定模板进行对话
func (cb *ChatBot) UseTemplate(templateName string, params map[string]interface{}, opts ...model.Option) error {
	tmpl, exists := cb.templates[templateName]
	if !exists {
		return fmt.Errorf("模板 '%s' 不存在", templateName)
	}

	// 格式化模板
	templateMessages, err := tmpl.Format(cb.ctx, params)
	if err != nil {
		return fmt.Errorf("模板格式化失败: %w", err)
	}

	// 组合消息：历史消息 + 模板消息（不将模板消息添加到历史）
	allMessages := append(cb.messages, templateMessages...)

	// 获取流式回复
	reader, err := cb.model.Stream(cb.ctx, allMessages, opts...)
	if err != nil {
		return fmt.Errorf("创建流式请求失败: %w", err)
	}
	defer reader.Close()

	// 处理流式内容
	var fullMessage *schema.Message
	for {
		chunk, err := reader.Recv()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return fmt.Errorf("接收流式数据失败: %w", err)
		}

		if fullMessage == nil {
			fullMessage = chunk
		} else {
			fullMessage, _ = schema.ConcatMessages([]*schema.Message{fullMessage, chunk})
		}

		if chunk.Content != "" {
			fmt.Print(chunk.Content)
		}
	}

	// 从模板消息中提取用户输入部分
	if len(templateMessages) > 0 {
		userInput := templateMessages[len(templateMessages)-1] // 最后一条通常是用户消息
		cb.messages = append(cb.messages, userInput)
	}

	if fullMessage != nil {
		cb.messages = append(cb.messages, fullMessage)
	}

	return nil
}

// Translate 使用翻译模板
func (cb *ChatBot) Translate(text, targetLang string) error {
	params := map[string]interface{}{
		"text":        text,
		"target_lang": targetLang,
	}

	fmt.Printf("翻译 %s -> %s: ", text, targetLang)
	return cb.UseTemplate("translate", params)
}

// GenerateCode 使用代码生成模板
func (cb *ChatBot) GenerateCode(requirement, language string) error {
	params := map[string]interface{}{
		"requirement": requirement,
		"language":    language,
	}

	fmt.Printf("生成 %s 代码: ", language)
	return cb.UseTemplate("code", params)
}

// Summarize 使用总结模板
func (cb *ChatBot) Summarize(content, style string, maxLength int) error {
	params := map[string]interface{}{
		"content":    content,
		"style":      style,
		"max_length": maxLength,
	}

	fmt.Printf("总结内容(%s风格): ", style)
	return cb.UseTemplate("summarize", params)
}

// AddDocument 添加文档到知识库
func (cb *ChatBot) AddDocument(id, content string) error {
	if cb.embedder == nil {
		return fmt.Errorf("嵌入器未初始化")
	}
	if cb.vectorstore == nil {
		return fmt.Errorf("向量存储未初始化")
	}
	if cb.textsplitter == nil {
		return fmt.Errorf("文本切分器未初始化")
	}

	// 1. 文本切分
	chunks := cb.textsplitter.Split(content)
	if len(chunks) == 0 {
		return fmt.Errorf("文本切分后没有有效分块")
	}

	// 2. 批量生成向量
	embeddings, err := cb.embedder.EmbedStrings(cb.ctx, chunks)
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
	if err := cb.vectorstore.InsertDocuments(cb.ctx, docs); err != nil {
		return fmt.Errorf("插入向量存储失败: %w", err)
	}

	debugLog("成功添加文档 %s，切分为 %d 个分块", id, len(chunks))
	return nil
}

// AddFile 导入单个文件
func (cb *ChatBot) AddFile(filePath string) (*fileimport.FileImportResult, error) {
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
	chunks := cb.textsplitter.Split(content)
	if len(chunks) == 0 {
		result.Error = fmt.Errorf("文本切分后没有有效分块")
		return result, result.Error
	}

	// 8. 批量生成向量
	embeddings, err := cb.embedder.EmbedStrings(cb.ctx, chunks)
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
	if err := cb.vectorstore.InsertDocuments(cb.ctx, docs); err != nil {
		result.Error = fmt.Errorf("插入向量存储失败: %w", err)
		return result, result.Error
	}

	result.DocID = docID
	result.ChunkCount = len(chunks)
	result.Success = true

	debugLog("成功添加文件: %s, 切分为 %d 个分块", docID, len(chunks))
	return result, nil
}

// AddDir 批量导入目录中的文件
func (cb *ChatBot) AddDir(dirPath string) error {
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

	debugLog("找到 %d 个文件待导入", len(files))

	// 3. 批量导入
	successCount := 0
	failCount := 0
	totalChunks := 0
	var failedFiles []string

	for _, file := range files {
		debugLog("处理文件: %s", file)

		result, err := cb.AddFile(file)
		if err != nil || !result.Success {
			debugLog("文件导入失败: %s, 错误: %v", file, err)
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
		if debugMode {
			for _, file := range failedFiles {
				fmt.Printf("  - %s\n", file)
			}
		}
	}

	return nil
}

// SearchDocuments 搜索相关文档
func (cb *ChatBot) SearchDocuments(query string, topK int) ([]*vectorstore.Document, error) {
	if cb.embedder == nil {
		return nil, fmt.Errorf("嵌入器未初始化")
	}
	if cb.vectorstore == nil {
		return nil, fmt.Errorf("向量存储未初始化")
	}

	// 生成查询向量
	embeddings, err := cb.embedder.EmbedStrings(cb.ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("生成查询向量失败: %w", err)
	}

	queryEmbedding := embeddings[0]

	// 使用 Milvus 搜索
	docs, err := cb.vectorstore.Search(cb.ctx, queryEmbedding, topK)
	if err != nil {
		return nil, fmt.Errorf("搜索失败: %w", err)
	}

	return docs, nil
}

// ChatWithRAG 使用 RAG 进行对话
func (cb *ChatBot) ChatWithRAG(query string, opts ...model.Option) error {
	// 搜索相关文档
	docs, err := cb.SearchDocuments(query, ragTopK)
	if err != nil {
		return fmt.Errorf("搜索文档失败: %w", err)
	}

	// 调试：打印检索结果
	debugLog("检索结果数量: %d", len(docs))
	for i, doc := range docs {
		debugLog("检索结果[%d]: doc_id=%s, chunk_index=%d, score=%.4f", i, doc.DocID, doc.ChunkIndex, doc.Score)
	}

	// 1. 分数阈值过滤
	var filteredDocs []*vectorstore.Document
	for _, doc := range docs {
		if doc.Score >= ragMinScore {
			filteredDocs = append(filteredDocs, doc)
		}
	}
	debugLog("过滤后结果数量: %d", len(filteredDocs))

	// 2. 无结果保护
	if len(filteredDocs) == 0 {
		fmt.Println("AI (RAG): 我不知道。")
		return nil
	}

	// 3. 第一层去重：按 doc_id + chunk_index
	uniqueDocs := make(map[string]*vectorstore.Document) // key: doc_id:chunk_index
	for _, doc := range filteredDocs {
		key := fmt.Sprintf("%s:%d", doc.DocID, doc.ChunkIndex)
		existing, exists := uniqueDocs[key]
		if !exists || doc.Score > existing.Score {
			uniqueDocs[key] = doc
		}
	}

	// 4. 第二层去重：按 content 兜底
	seenContent := make(map[string]bool)
	var dedupedDocs []*vectorstore.Document
	for _, doc := range uniqueDocs {
		if !seenContent[doc.Content] {
			seenContent[doc.Content] = true
			dedupedDocs = append(dedupedDocs, doc)
		}
	}
	debugLog("双层去重后结果数量: %d", len(dedupedDocs))

	// 5. 按分数降序排序
	sort.Slice(dedupedDocs, func(i, j int) bool {
		return dedupedDocs[i].Score > dedupedDocs[j].Score
	})

	// 6. 构建上下文（限制长度和 chunk 数）
	contextBuilder := strings.Builder{}
	totalLen := 0
	chunkCount := 0
	for _, doc := range dedupedDocs {
		content := doc.Content + "\n"

		// 检查 chunk 数限制
		if chunkCount >= ragMaxContextChunks {
			debugLog("达到最大 chunk 数 %d，停止拼接", ragMaxContextChunks)
			break
		}

		// 检查长度限制
		if totalLen+len(content) > ragMaxContextLen {
			debugLog("上下文达到最大长度 %d，停止拼接", ragMaxContextLen)
			break
		}

		contextBuilder.WriteString(content)
		totalLen += len(content)
		chunkCount++
	}
	conText := contextBuilder.String()

	debugLog("上下文总长度: %d 字符, chunk 数: %d", totalLen, chunkCount)

	// 7. 构建优化后的 promptWord（不包含【上下文】标签）
	promptWord := fmt.Sprintf(`你是一个基于知识库的问答助手，请严格根据上下文回答。

【重要要求】
1. 必须完整列出上下文中的全部关键信息
2. 不要遗漏任何相关信息
3. 如果上下文中有多个相关点，请全部列出
4. 如果上下文中包含多个答案，必须完整列出全部答案，不要遗漏
5. 严格基于上下文回答，不要编造信息

【上下文】
%s

【问题】
%s

如果上下文中没有答案，请明确说不知道。`, conText, query)

	// 调试：打印完整 promptWord
	debugLog("RAG Prompt: %s", promptWord)

	// 保存当前消息历史长度（避免污染）
	prevMessageCount := len(cb.messages)

	// 添加临时 RAG promptWord（不污染长期历史）
	cb.messages = append(cb.messages, schema.UserMessage(promptWord))

	// 获取流式回复
	reader, err := cb.model.Stream(cb.ctx, cb.messages, opts...)
	if err != nil {
		return fmt.Errorf("创建流式请求失败: %w", err)
	}
	defer reader.Close()

	// 处理流式内容
	var fullMessage *schema.Message
	for {
		chunk, err := reader.Recv()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return fmt.Errorf("接收流式数据失败: %w", err)
		}

		// 合并消息碎片
		if fullMessage == nil {
			fullMessage = chunk
		} else {
			fullMessage, _ = schema.ConcatMessages([]*schema.Message{fullMessage, chunk})
		}

		// 实时输出
		if chunk.Content != "" {
			fmt.Print(chunk.Content)
		}
	}

	// 恢复消息历史（移除临时 RAG promptWord）
	cb.messages = cb.messages[:prevMessageCount]

	// 只添加用户原问题和最终回答到历史
	cb.messages = append(cb.messages, schema.UserMessage(query))
	if fullMessage != nil {
		cb.messages = append(cb.messages, fullMessage)
	}

	return nil
}

// ChatStream 进行流式对话
func (cb *ChatBot) ChatStream(userInput string, opts ...model.Option) error {
	// 添加用户消息
	cb.messages = append(cb.messages, schema.UserMessage(userInput))

	// 获取流式回复
	reader, err := cb.model.Stream(cb.ctx, cb.messages, opts...)
	if err != nil {
		return fmt.Errorf("创建流式请求失败: %w", err)
	}
	defer reader.Close() // 注意要关闭

	// 处理流式内容
	var fullMessage *schema.Message
	for {
		chunk, err := reader.Recv()
		if err != nil {
			if err.Error() == "EOF" {
				break // 正常结束
			}
			return fmt.Errorf("接收流式数据失败: %w", err)
		}

		// 合并消息碎片
		if fullMessage == nil {
			fullMessage = chunk
		} else {
			fullMessage, _ = schema.ConcatMessages([]*schema.Message{fullMessage, chunk})
		}

		// 实时输出
		if chunk.Content != "" {
			fmt.Print(chunk.Content)
		}
	}

	// 将完整的响应消息添加到历史记录
	if fullMessage != nil {
		cb.messages = append(cb.messages, fullMessage)
	}

	return nil
}

func main() {
	// 1. 创建上下文
	ctx := context.Background()

	// 2. 创建 ChatModel (使用豆包 Ark)
	timeout := 30 * time.Second
	chatModel, err := arkModel.NewChatModel(ctx, &arkModel.ChatModelConfig{
		APIKey:  os.Getenv("ARK_API_KEY"),
		Model:   os.Getenv("ARK_MODEL_NAME"),
		Timeout: &timeout,
	})
	if err != nil {
		log.Fatalf("创建 ChatModel 失败: %v", err)
	}

	// 3. 创建嵌入器 (用于 RAG)
	var embedder embedding.Embedder
	if embedderModel := os.Getenv("EMBEDDER"); embedderModel != "" {
		// 使用火山引擎的嵌入器
		apiKey := os.Getenv("ARK_EMBEDDER_API_KEY")
		if apiKey == "" {
			log.Printf("未设置 ARK_EMBEDDER_API_KEY 环境变量，RAG功能将不可用")
			embedder = nil
		} else {
			embedder = NewVolcengineEmbedder(apiKey, embedderModel)
			fmt.Println("RAG 功能已启用（使用火山引擎嵌入器）")
		}
	} else {
		fmt.Println("未设置 EMBEDDER 环境变量，RAG 功能将不可用")
	}

	// 4. 创建 Milvus 向量存储
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
			debugLog("Milvus 向量存储已启用 (address=%s, dim=%d, topK=%d)", milvusAddress, milvusDim, milvusTopK)
		}
	} else {
		vs = nil
	}

	// 5. 创建文本切分器
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
		debugLog("文本切分器已启用 (chunk_size=%d, chunk_overlap=%d)", chunkSize, chunkOverlap)
	} else {
		splitter = nil
	}

	// 6. 创建聊天机器人（使用 BaseChatModel 接口）
	chatBot := NewChatBot(ctx, chatModel, embedder, vs, splitter)

	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("=== Eino ChatBot with Templates & RAG (Milvus) ===")
	fmt.Println("可用命令：")
	fmt.Println("  /translate <文本> <目标语言>      - 翻译")
	fmt.Println("  /code <需求> <语言>               - 代码生成")
	fmt.Println("  /summarize <内容> <风格> <长度>    - 内容总结")
	fmt.Println("  /add <文档内容>                   - 添加文档到知识库")
	fmt.Println("  /add_file <文件路径>              - 导入单个文件（支持 .txt 和 .md）")
	fmt.Println("  /add_dir <目录路径>               - 批量导入目录中的文件")
	fmt.Println("  /rag <问题>                       - 使用 RAG 回答问题")
	fmt.Println("  其他输入                           - 普通对话")
	fmt.Println("  exit                              - 退出")
	fmt.Println("=====================================")
	fmt.Println("提示：带空格的路径请使用引号，例如：/add_file \"./data/my file.txt\"")
	fmt.Println("注意：重复导入会创建新的向量记录，由检索层通过相似度去重")

	docCounter := 0
	for {
		fmt.Print("\n你: ")
		if !scanner.Scan() {
			break
		}

		userInput := strings.TrimSpace(scanner.Text())
		if userInput == "exit" {
			fmt.Println("再见！")
			break
		}

		if userInput == "" {
			continue
		}

		// 解析命令
		if strings.HasPrefix(userInput, "/") {
			parts := strings.Fields(userInput)
			command := parts[0]

			switch command {
			case "/translate":
				if len(parts) >= 3 {
					text := strings.Join(parts[1:len(parts)-1], " ")
					targetLang := parts[len(parts)-1]
					err := chatBot.Translate(text, targetLang)
					if err != nil {
						log.Printf("翻译失败: %v", err)
					}
				} else {
					fmt.Println("用法: /translate <文本> <目标语言>")
				}

			case "/code":
				if len(parts) >= 3 {
					requirement := strings.Join(parts[1:len(parts)-1], " ")
					language := parts[len(parts)-1]
					err := chatBot.GenerateCode(requirement, language)
					if err != nil {
						log.Printf("代码生成失败: %v", err)
					}
				} else {
					fmt.Println("用法: /code <需求> <语言>")
				}

			case "/summarize":
				if len(parts) >= 3 {
					content := strings.Join(parts[1:len(parts)-2], " ")
					style := parts[len(parts)-2]
					maxLength := 100 // 默认值
					if len(parts) > 3 {
						fmt.Sscanf(parts[len(parts)-1], "%d", &maxLength)
					}
					err := chatBot.Summarize(content, style, maxLength)
					if err != nil {
						log.Printf("总结失败: %v", err)
					}
				} else {
					fmt.Println("用法: /summarize <内容> <风格> <长度>")
				}

			case "/add":
				if len(parts) > 1 {
					content := strings.Join(parts[1:], " ")
					docCounter++
					docID := fmt.Sprintf("doc_%d", docCounter)
					err := chatBot.AddDocument(docID, content)
					if err != nil {
						log.Printf("添加文档失败: %v", err)
						docCounter-- // 失败则回退计数
					} else {
						fmt.Printf("成功添加文档: %s\n", docID)
						debugLog("文档内容: %s", content)
					}
				} else {
					fmt.Println("用法: /add <文档内容>")
				}

			case "/add_file":
				if len(parts) > 1 {
					filePath := strings.Join(parts[1:], " ")
					result, err := chatBot.AddFile(filePath)
					if err != nil {
						log.Printf("文件导入失败: %v", err)
					} else if result.Success {
						fmt.Printf("成功导入文件: %s (%d 个分块)\n", result.DocID, result.ChunkCount)
					}
				} else {
					fmt.Println("用法: /add_file <文件路径>")
					fmt.Println("注意：带空格的路径请使用引号，例如：/add_file \"./data/my file.txt\"")
				}

			case "/add_dir":
				if len(parts) > 1 {
					dirPath := strings.Join(parts[1:], " ")
					err := chatBot.AddDir(dirPath)
					if err != nil {
						log.Printf("目录导入失败: %v", err)
					}
				} else {
					fmt.Println("用法: /add_dir <目录路径>")
					fmt.Println("注意：带空格的路径请使用引号，例如：/add_dir \"./my data/\"")
				}

			case "/rag":
				if len(parts) > 1 {
					query := strings.Join(parts[1:], " ")
					fmt.Print("AI (RAG): ")
					err := chatBot.ChatWithRAG(query)
					if err != nil {
						log.Printf("RAG对话失败: %v", err)
					}
				} else {
					fmt.Println("用法: /rag <问题>")
				}

			default:
				fmt.Printf("未知命令: %s\n", command)
			}
		} else {
			// 普通对话
			fmt.Print("AI: ")
			err := chatBot.ChatStream(userInput)
			if err != nil {
				log.Printf("对话失败: %v", err)
				continue
			}
		}

		fmt.Println()
	}
}
