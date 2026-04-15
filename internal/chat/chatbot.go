package chat

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/schema"
	"github.com/volcengine/volcengine-go-sdk/service/arkruntime"
	arkruntimeModel "github.com/volcengine/volcengine-go-sdk/service/arkruntime/model"

	"eino-tutorial/internal/retrieval"
	"eino-tutorial/internal/utils"
	"eino-tutorial/internal/vectorstore"
)

// ChatBot 聊天机器人结构体
type ChatBot struct {
	model               model.BaseChatModel
	ctx                 context.Context
	messages            []*schema.Message
	templates           map[string]prompt.ChatTemplate
	retrievalService    *retrieval.Service
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
func NewChatBot(ctx context.Context, chatModel model.BaseChatModel, retrievalService *retrieval.Service, ragMinScore float64, ragTopK int, ragMaxContextLen int, ragMaxContextChunks int) *ChatBot {
	bot := &ChatBot{
		model: chatModel,
		ctx:   ctx,
		messages: []*schema.Message{
			schema.SystemMessage("你是一个友好的 AI 助手"),
		},
		templates:           make(map[string]prompt.ChatTemplate),
		retrievalService:    retrievalService,
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

// getDocID 获取文档 ID
func (cb *ChatBot) getDocID(doc *vectorstore.Document) string {
	return doc.DocID
}

// getChunkIndex 获取分块索引
func (cb *ChatBot) getChunkIndex(doc *vectorstore.Document) int64 {
	return doc.ChunkIndex
}

// getScore 获取相似度分数
func (cb *ChatBot) getScore(doc *vectorstore.Document) float64 {
	return doc.Score
}

// getContent 获取文档内容
func (cb *ChatBot) getContent(doc *vectorstore.Document) string {
	return doc.Content
}

// getDocDedupKey 生成文档去重 key
func (cb *ChatBot) getDocDedupKey(doc *vectorstore.Document) string {
	return fmt.Sprintf("%s:%d", cb.getDocID(doc), cb.getChunkIndex(doc))
}

// ChatWithRAG 使用 RAG 进行对话
func (cb *ChatBot) ChatWithRAG(query string, opts ...model.Option) error {
	// 检查检索服务是否可用
	if cb.retrievalService == nil {
		fmt.Println("AI (RAG): 知识库检索功能未启用，无法使用 RAG。请检查 EMBEDDER、ARK_EMBEDDER_API_KEY、MILVUS_ADDRESS 等配置。")
		return nil
	}

	// 搜索相关文档
	docs, err := cb.retrievalService.SearchDocuments(query, cb.ragTopK)
	if err != nil {
		return fmt.Errorf("搜索文档失败: %w", err)
	}

	// 调试：打印检索结果
	utils.DebugLog("检索结果数量: %d", len(docs))
	for i, doc := range docs {
		utils.DebugLog("检索结果[%d]: doc_id=%s, chunk_index=%d, score=%.4f", i, cb.getDocID(doc), cb.getChunkIndex(doc), cb.getScore(doc))
	}

	// 1. 分数阈值过滤
	var filteredDocs []*vectorstore.Document
	for _, doc := range docs {
		if cb.getScore(doc) >= cb.ragMinScore {
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
		key := cb.getDocDedupKey(doc)
		existing, exists := uniqueDocs[key]
		if !exists || cb.getScore(doc) > cb.getScore(existing) {
			uniqueDocs[key] = doc
		}
	}

	// 4. 第二层去重：按 content 兜底
	seenContent := make(map[string]bool)
	var dedupedDocs []*vectorstore.Document
	for _, doc := range uniqueDocs {
		if !seenContent[cb.getContent(doc)] {
			seenContent[cb.getContent(doc)] = true
			dedupedDocs = append(dedupedDocs, doc)
		}
	}
	utils.DebugLog("双层去重后结果数量: %d", len(dedupedDocs))

	// 5. 按分数降序排序
	sort.Slice(dedupedDocs, func(i, j int) bool {
		return cb.getScore(dedupedDocs[i]) > cb.getScore(dedupedDocs[j])
	})

	// 6. 构建上下文（限制长度和 chunk 数）
	contextBuilder := strings.Builder{}
	totalLen := 0
	chunkCount := 0
	for _, doc := range dedupedDocs {
		content := cb.getContent(doc) + "\n"

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
