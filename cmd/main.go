package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
	"time"

	arkModel "github.com/cloudwego/eino-ext/components/model/ark"
	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/schema"
	"github.com/volcengine/volcengine-go-sdk/service/arkruntime"
	arkruntimeModel "github.com/volcengine/volcengine-go-sdk/service/arkruntime/model"
)

// ChatBot 结构体，使用 eino BaseChatModel 接口和 ChatTemplate
type ChatBot struct {
	model     model.BaseChatModel
	ctx       context.Context
	messages  []*schema.Message
	templates map[string]prompt.ChatTemplate
	embedder  embedding.Embedder // 嵌入器用于 RAG
	documents []*Document        // 文档存储
}

// Document 文档结构
type Document struct {
	ID        string
	Content   string
	Embedding []float64
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
func NewChatBot(ctx context.Context, chatModel model.BaseChatModel, embedder embedding.Embedder) *ChatBot {
	bot := &ChatBot{
		model: chatModel,
		ctx:   ctx,
		messages: []*schema.Message{
			schema.SystemMessage("你是一个友好的 AI 助手"),
		},
		templates: make(map[string]prompt.ChatTemplate),
		embedder:  embedder,
		documents: make([]*Document, 0),
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

	// 生成文档的向量
	embeddings, err := cb.embedder.EmbedStrings(cb.ctx, []string{content})

	if err != nil {
		return fmt.Errorf("生成文档向量失败: %w", err)
	}

	doc := &Document{
		ID:        id,
		Content:   content,
		Embedding: embeddings[0],
	}

	cb.documents = append(cb.documents, doc)
	return nil
}

// cosineSimilarity 计算余弦相似度
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// SearchDocuments 搜索相关文档
func (cb *ChatBot) SearchDocuments(query string, topK int) ([]*Document, error) {
	if cb.embedder == nil {
		return nil, fmt.Errorf("嵌入器未初始化")
	}

	// 生成查询向量
	embeddings, err := cb.embedder.EmbedStrings(cb.ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("生成查询向量失败: %w", err)
	}

	queryEmbedding := embeddings[0]

	// 计算相似度
	type docScore struct {
		doc   *Document
		score float64
	}

	var scores []docScore
	for _, doc := range cb.documents {
		score := cosineSimilarity(queryEmbedding, doc.Embedding)
		scores = append(scores, docScore{doc: doc, score: score})
	}

	// 排序并返回 topK
	for i := 0; i < len(scores); i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[i].score < scores[j].score {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}

	if topK > len(scores) {
		topK = len(scores)
	}

	result := make([]*Document, topK)
	for i := 0; i < topK; i++ {
		result[i] = scores[i].doc
	}

	return result, nil
}

// ChatWithRAG 使用 RAG 进行对话
func (cb *ChatBot) ChatWithRAG(query string, opts ...model.Option) error {
	// 搜索相关文档
	docs, err := cb.SearchDocuments(query, 3)
	if err != nil {
		return fmt.Errorf("搜索文档失败: %w", err)
	}

	// 构建上下文
	conText := ""
	if len(docs) > 0 {
		conText = "参考文档：\n"
		for i, doc := range docs {
			conText += fmt.Sprintf("%d. %s\n", i+1, doc.Content)
		}
		conText += "\n"
	}

	// 构建增强的用户消息
	enhancedQuery := conText + "用户问题：" + query

	// 添加用户消息
	cb.messages = append(cb.messages, schema.UserMessage(enhancedQuery))

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

		if fullMessage == nil {
			fullMessage = chunk
		} else {
			fullMessage, _ = schema.ConcatMessages([]*schema.Message{fullMessage, chunk})
		}

		if chunk.Content != "" {
			fmt.Print(chunk.Content)
		}
	}

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

	// 4. 创建聊天机器人（使用 BaseChatModel 接口）
	chatBot := NewChatBot(ctx, chatModel, embedder)

	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("=== Eino ChatBot with Templates & RAG ===")
	fmt.Println("可用命令：")
	fmt.Println("  /translate <文本> <目标语言>      - 翻译")
	fmt.Println("  /code <需求> <语言>               - 代码生成")
	fmt.Println("  /summarize <内容> <风格> <长度>    - 内容总结")
	fmt.Println("  /add <文档内容>                   - 添加文档到知识库")
	fmt.Println("  /rag <问题>                       - 使用 RAG 回答问题")
	fmt.Println("  其他输入                           - 普通对话")
	fmt.Println("  exit                              - 退出")
	fmt.Println("=====================================")

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
					docID := fmt.Sprintf("doc_%d", len(chatBot.documents)+1)
					err := chatBot.AddDocument(docID, content)
					if err != nil {
						log.Printf("添加文档失败: %v", err)
					} else {
						fmt.Printf("成功添加文档: %s\n", docID)
					}
				} else {
					fmt.Println("用法: /add <文档内容>")
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
