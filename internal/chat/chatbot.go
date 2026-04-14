package chat

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/schema"
	"github.com/google/uuid"
	"github.com/volcengine/volcengine-go-sdk/service/arkruntime"
	arkruntimeModel "github.com/volcengine/volcengine-go-sdk/service/arkruntime/model"

	"eino-tutorial/internal/fileimport"
	"eino-tutorial/internal/textsplitter"
	"eino-tutorial/internal/utils"
	"eino-tutorial/internal/vectorstore"
)

// ChatBot 聊天机器人结构体
type ChatBot struct {
	model               model.BaseChatModel
	ctx                 context.Context
	messages            []*schema.Message
	templates           map[string]prompt.ChatTemplate
	embedder            embedding.Embedder
	vectorstore         vectorstore.VectorStore
	textsplitter        *textsplitter.TextSplitter
	ragMinScore         float64
	ragTopK             int
	ragMaxContextLen    int
	ragMaxContextChunks int
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
func NewChatBot(ctx context.Context, chatModel model.BaseChatModel, embedder embedding.Embedder, vs vectorstore.VectorStore, ts *textsplitter.TextSplitter, ragMinScore float64, ragTopK int, ragMaxContextLen int, ragMaxContextChunks int) *ChatBot {
	bot := &ChatBot{
		model: chatModel,
		ctx:   ctx,
		messages: []*schema.Message{
			schema.SystemMessage("你是一个友好的 AI 助手"),
		},
		templates:           make(map[string]prompt.ChatTemplate),
		embedder:            embedder,
		vectorstore:         vs,
		textsplitter:        ts,
		ragMinScore:         ragMinScore,
		ragTopK:             ragTopK,
		ragMaxContextLen:    ragMaxContextLen,
		ragMaxContextChunks: ragMaxContextChunks,
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

	utils.DebugLog("成功添加文档 %s，切分为 %d 个分块", id, len(chunks))
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

	utils.DebugLog("成功添加文件: %s, 切分为 %d 个分块", docID, len(chunks))
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

	utils.DebugLog("找到 %d 个文件待导入", len(files))

	// 3. 批量导入
	successCount := 0
	failCount := 0
	totalChunks := 0
	var failedFiles []string

	for _, file := range files {
		utils.DebugLog("处理文件: %s", file)

		result, err := cb.AddFile(file)
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
	docs, err := cb.SearchDocuments(query, cb.ragTopK)
	if err != nil {
		return fmt.Errorf("搜索文档失败: %w", err)
	}

	// 调试：打印检索结果
	utils.DebugLog("检索结果数量: %d", len(docs))
	for i, doc := range docs {
		utils.DebugLog("检索结果[%d]: doc_id=%s, chunk_index=%d, score=%.4f", i, doc.DocID, doc.ChunkIndex, doc.Score)
	}

	// 1. 分数阈值过滤
	var filteredDocs []*vectorstore.Document
	for _, doc := range docs {
		if doc.Score >= cb.ragMinScore {
			filteredDocs = append(filteredDocs, doc)
		}
	}
	utils.DebugLog("过滤后结果数量: %d", len(filteredDocs))

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
	utils.DebugLog("双层去重后结果数量: %d", len(dedupedDocs))

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
		if chunkCount >= cb.ragMaxContextChunks {
			utils.DebugLog("达到最大 chunk 数 %d，停止拼接", cb.ragMaxContextChunks)
			break
		}

		// 检查长度限制
		if totalLen+len(content) > cb.ragMaxContextLen {
			utils.DebugLog("上下文达到最大长度 %d，停止拼接", cb.ragMaxContextLen)
			break
		}

		contextBuilder.WriteString(content)
		totalLen += len(content)
		chunkCount++
	}
	conText := contextBuilder.String()

	utils.DebugLog("上下文总长度: %d 字符, chunk 数: %d", totalLen, chunkCount)

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
	utils.DebugLog("RAG Prompt: %s", promptWord)

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
