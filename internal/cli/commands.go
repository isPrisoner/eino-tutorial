package cli

import (
	"fmt"
	"log"
	"strings"

	"eino-tutorial/internal/utils"
)

// handleTranslate 处理翻译命令
func handleTranslate(h *Handler, parts []string) error {
	if len(parts) < 3 {
		fmt.Println("用法: /translate <文本> <目标语言>")
		return nil
	}

	text := strings.Join(parts[1:len(parts)-1], " ")
	targetLang := parts[len(parts)-1]
	err := h.chatBot.Translate(text, targetLang)
	if err != nil {
		return fmt.Errorf("翻译失败: %w", err)
	}
	return nil
}

// handleCode 处理代码生成命令
func handleCode(h *Handler, parts []string) error {
	if len(parts) < 3 {
		fmt.Println("用法: /code <需求> <语言>")
		return nil
	}

	requirement := strings.Join(parts[1:len(parts)-1], " ")
	language := parts[len(parts)-1]
	err := h.chatBot.GenerateCode(requirement, language)
	if err != nil {
		return fmt.Errorf("代码生成失败: %w", err)
	}
	return nil
}

// handleSummarize 处理总结命令
func handleSummarize(h *Handler, parts []string) error {
	if len(parts) < 3 {
		fmt.Println("用法: /summarize <内容> <风格> <长度>")
		return nil
	}

	content := strings.Join(parts[1:len(parts)-2], " ")
	style := parts[len(parts)-2]
	maxLength := 100 // 默认值
	if len(parts) > 3 {
		fmt.Sscanf(parts[len(parts)-1], "%d", &maxLength)
	}
	err := h.chatBot.Summarize(content, style, maxLength)
	if err != nil {
		return fmt.Errorf("总结失败: %w", err)
	}
	return nil
}

// handleAdd 处理添加文档命令
func handleAdd(h *Handler, parts []string) error {
	if h.ingestService == nil {
		fmt.Println("知识库入库功能未启用，无法添加文档。请检查 EMBEDDER、ARK_EMBEDDER_API_KEY、MILVUS_ADDRESS 等配置。")
		return nil
	}

	if len(parts) <= 1 {
		fmt.Println("用法: /add <文档内容>")
		return nil
	}

	content := strings.Join(parts[1:], " ")
	h.docCounter++
	docID := fmt.Sprintf("doc_%d", h.docCounter)
	err := h.ingestService.AddDocument(docID, content)
	if err != nil {
		log.Printf("添加文档失败: %v", err)
		h.docCounter-- // 失败则回退计数
		return fmt.Errorf("添加文档失败: %w", err)
	}
	fmt.Printf("成功添加文档: %s\n", docID)
	utils.DebugLog("文档内容: %s", content)
	return nil
}

// handleAddFile 处理添加文件命令
func handleAddFile(h *Handler, parts []string) error {
	if h.ingestService == nil {
		fmt.Println("知识库入库功能未启用，无法添加文档。请检查 EMBEDDER、ARK_EMBEDDER_API_KEY、MILVUS_ADDRESS 等配置。")
		return nil
	}

	if len(parts) <= 1 {
		fmt.Println("用法: /add_file <文件路径>")
		fmt.Println("注意：带空格的路径请使用引号，例如：/add_file \"./data/my file.txt\"")
		return nil
	}

	filePath := strings.Join(parts[1:], " ")
	result, err := h.ingestService.AddFile(filePath)
	if err != nil {
		log.Printf("文件导入失败: %v", err)
		return fmt.Errorf("文件导入失败: %w", err)
	}
	if result.Success {
		fmt.Printf("成功导入文件: %s (%d 个分块)\n", result.DocID, result.ChunkCount)
	}
	return nil
}

// handleAddDir 处理添加目录命令
func handleAddDir(h *Handler, parts []string) error {
	if h.ingestService == nil {
		fmt.Println("知识库入库功能未启用，无法添加文档。请检查 EMBEDDER、ARK_EMBEDDER_API_KEY、MILVUS_ADDRESS 等配置。")
		return nil
	}

	if len(parts) <= 1 {
		fmt.Println("用法: /add_dir <目录路径>")
		fmt.Println("注意：带空格的路径请使用引号，例如：/add_dir \"./my data/\"")
		return nil
	}

	dirPath := strings.Join(parts[1:], " ")
	err := h.ingestService.AddDir(dirPath)
	if err != nil {
		log.Printf("目录导入失败: %v", err)
		return fmt.Errorf("目录导入失败: %w", err)
	}
	return nil
}

// handleRAG 处理 RAG 对话命令
func handleRAG(h *Handler, parts []string) error {
	if len(parts) <= 1 {
		fmt.Println("用法: /rag <问题>")
		return nil
	}

	query := strings.Join(parts[1:], " ")
	fmt.Print("AI (RAG): ")
	err := h.chatBot.ChatWithRAG(query)
	if err != nil {
		log.Printf("RAG对话失败: %v", err)
		return fmt.Errorf("RAG对话失败: %w", err)
	}
	return nil
}

// handleNormalChat 处理普通对话
func handleNormalChat(h *Handler, input string) error {
	fmt.Print("AI: ")
	err := h.chatBot.ChatStream(input)
	if err != nil {
		log.Printf("对话失败: %v", err)
		return fmt.Errorf("对话失败: %w", err)
	}
	return nil
}
