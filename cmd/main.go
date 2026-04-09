package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/cloudwego/eino-ext/components/model/ark"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/schema"
)

// ChatBot 结构体，使用 eino BaseChatModel 接口和 ChatTemplate
type ChatBot struct {
	model     model.BaseChatModel
	ctx       context.Context
	messages  []*schema.Message
	templates map[string]prompt.ChatTemplate
}

// NewChatBot 创建新的聊天机器人实例
func NewChatBot(ctx context.Context, chatModel model.BaseChatModel) *ChatBot {
	bot := &ChatBot{
		model: chatModel,
		ctx:   ctx,
		messages: []*schema.Message{
			schema.SystemMessage("你是一个友好的 AI 助手"),
		},
		templates: make(map[string]prompt.ChatTemplate),
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
	arkModel, err := ark.NewChatModel(ctx, &ark.ChatModelConfig{
		APIKey:  os.Getenv("ARK_API_KEY"),
		Model:   os.Getenv("ARK_MODEL_NAME"),
		Timeout: &timeout,
	})
	if err != nil {
		log.Fatalf("创建 ChatModel 失败: %v", err)
	}

	// 3. 创建聊天机器人（使用 BaseChatModel 接口）
	chatBot := NewChatBot(ctx, arkModel)

	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("=== Eino ChatBot with Templates ===")
	fmt.Println("可用命令：")
	fmt.Println("  /translate <文本> <目标语言>  - 翻译")
	fmt.Println("  /code <需求> <语言>           - 代码生成")
	fmt.Println("  /summarize <内容> <风格> <长度> - 内容总结")
	fmt.Println("  其他输入                       - 普通对话")
	fmt.Println("  exit                          - 退出")
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
