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
	"github.com/cloudwego/eino/schema"
)

// ChatBot 结构体，使用 eino BaseChatModel 接口
type ChatBot struct {
	model    model.BaseChatModel
	ctx      context.Context
	messages []*schema.Message
}

// NewChatBot 创建新的聊天机器人实例
func NewChatBot(ctx context.Context, chatModel model.BaseChatModel) *ChatBot {
	return &ChatBot{
		model: chatModel,
		ctx:   ctx,
		messages: []*schema.Message{
			schema.SystemMessage("你是一个友好的 AI 助手"),
		},
	}
}

// ChatGenerate 进行非流式对话
func (cb *ChatBot) ChatGenerate(userInput string, opts ...model.Option) (*schema.Message, error) {
	// 添加用户消息
	cb.messages = append(cb.messages, schema.UserMessage(userInput))

	// 生成回复
	response, err := cb.model.Generate(cb.ctx, cb.messages, opts...)
	if err != nil {
		return nil, fmt.Errorf("生成响应失败: %w", err)
	}

	// 添加助手响应到历史记录
	cb.messages = append(cb.messages, response)

	return response, nil
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
	fmt.Println("对话开始（输入'exit'退出会话）：")

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

		// 使用流式对话
		fmt.Print("AI: ")
		err := chatBot.ChatStream(userInput)

		if err != nil {
			log.Printf("对话失败: %v", err)
			continue
		}

		fmt.Println() // 换行
	}
}
