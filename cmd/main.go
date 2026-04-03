package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/cloudwego/eino-ext/components/model/ark"
	"github.com/cloudwego/eino/schema"
)

func main() {
	// 1. 创建上下文
	ctx := context.Background()

	// 2. 创建 ChatModel (使用豆包 Ark)
	chatModel, err := ark.NewChatModel(ctx, &ark.ChatModelConfig{
		APIKey: os.Getenv("ARK_API_KEY"),
		Model:  os.Getenv("ARK_MODEL_NAME"),
	})
	if err != nil {
		log.Fatalf("创建 ChatModel 失败: %v", err)
	}

	// 3. 准备消息
	messages := []*schema.Message{
		schema.SystemMessage("你是一个友好的 AI 助手"),
	}

	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("对话开始（输入'exit'退出会话）：")

	for {
		fmt.Print("你: ")
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

		// 添加用户消息
		messages = append(messages, schema.UserMessage(userInput))

		// 生成 AI 响应（流式输出）
		var fullMessage *schema.Message

		// 使用流式输出
		streamReader, err := chatModel.Stream(ctx, messages)
		if err != nil {
			log.Printf("创建流式请求失败: %v", err)
			continue
		}

		// 读取流式响应，收到第一个 chunk 时才打印 "AI: " 前缀
		firstChunk := true
		for {
			chunk, err := streamReader.Recv()
			if err != nil {
				if err.Error() == "EOF" {
					break // 正常结束
				}
				log.Printf("接收流式数据失败: %v", err)
				break
			}

			// 合并消息碎片
			if fullMessage == nil {
				fullMessage = chunk
			} else {
				fullMessage, _ = schema.ConcatMessages([]*schema.Message{fullMessage, chunk})
			}

			if chunk.Content != "" {
				if firstChunk {
					fmt.Print("AI: ")
					firstChunk = false
				}
				fmt.Print(chunk.Content) // 实时输出
			}
		}

		fmt.Print("\n\n") // 换行

		// 将完整的响应消息添加到历史记录
		if fullMessage != nil {
			messages = append(messages, fullMessage)
		}
	}

}
