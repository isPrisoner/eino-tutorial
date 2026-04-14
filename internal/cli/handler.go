package cli

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"eino-tutorial/internal/chat"
)

// Handler CLI 命令处理器
type Handler struct {
	chatBot    *chat.ChatBot
	scanner    *bufio.Scanner
	docCounter int
}

// NewHandler 创建 CLI 处理器
func NewHandler(chatBot *chat.ChatBot) *Handler {
	return &Handler{
		chatBot:    chatBot,
		scanner:    bufio.NewScanner(os.Stdin),
		docCounter: 0,
	}
}

// Run 启动 CLI 主循环
func (h *Handler) Run() error {
	h.printHelp()

	for {
		fmt.Print("\n你: ")
		if !h.scanner.Scan() {
			break
		}

		userInput := strings.TrimSpace(h.scanner.Text())
		if userInput == "exit" {
			fmt.Println("再见！")
			break
		}

		if userInput == "" {
			continue
		}

		// 处理命令
		if err := h.handleCommand(userInput); err != nil {
			fmt.Printf("错误: %v\n", err)
		}

		fmt.Println()
	}

	return nil
}

// printHelp 打印帮助信息
func (h *Handler) printHelp() {
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
}

// handleCommand 处理用户输入命令
func (h *Handler) handleCommand(input string) error {
	// 解析命令
	if strings.HasPrefix(input, "/") {
		parts := strings.Fields(input)
		if len(parts) == 0 {
			return fmt.Errorf("无效命令")
		}
		command := parts[0]

		switch command {
		case "/translate":
			return handleTranslate(h, parts)
		case "/code":
			return handleCode(h, parts)
		case "/summarize":
			return handleSummarize(h, parts)
		case "/add":
			return handleAdd(h, parts)
		case "/add_file":
			return handleAddFile(h, parts)
		case "/add_dir":
			return handleAddDir(h, parts)
		case "/rag":
			return handleRAG(h, parts)
		default:
			fmt.Printf("未知命令: %s\n", command)
			return nil
		}
	} else {
		// 普通对话
		return handleNormalChat(h, input)
	}
}
