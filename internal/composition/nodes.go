package composition

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"

	"eino-tutorial/internal/retrieval"
	"eino-tutorial/internal/tools"
	"eino-tutorial/internal/utils"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"
)

// RetrievalNode 检索节点
func RetrievalNode(retrievalService *retrieval.Service) func(ctx context.Context, state *RAGGraphState) (*RAGGraphState, error) {
	return func(ctx context.Context, state *RAGGraphState) (*RAGGraphState, error) {
		// 搜索相关文档
		docs, err := retrievalService.Retrieve(ctx, state.UserInput, retriever.WithTopK(5))
		if err != nil {
			return nil, fmt.Errorf("搜索文档失败: %w", err)
		}

		utils.DebugLog("检索结果数量: %d", len(docs))

		// 1. 分数阈值过滤
		var filteredDocs []*schema.Document
		for _, doc := range docs {
			score := getScore(doc)
			if score >= 0.5 {
				filteredDocs = append(filteredDocs, doc)
			}
		}
		utils.DebugLog("过滤后结果数量: %d", len(filteredDocs))

		// 2. 无结果保护
		if len(filteredDocs) == 0 {
			state.RetrievedContext = ""
			return state, nil
		}

		// 3. 第一层去重：按 doc_id + chunk_index
		uniqueDocs := make(map[string]*schema.Document)
		for _, doc := range filteredDocs {
			key := getDocDedupKey(doc)
			existing, exists := uniqueDocs[key]
			score := getScore(doc)
			if !exists || score > getScore(existing) {
				uniqueDocs[key] = doc
			}
		}

		// 4. 第二层去重：按 content 兜底
		seenContent := make(map[string]bool)
		var dedupedDocs []*schema.Document
		for _, doc := range uniqueDocs {
			content := getContent(doc)
			if !seenContent[content] {
				seenContent[content] = true
				dedupedDocs = append(dedupedDocs, doc)
			}
		}

		// 5. 按分数降序排序
		sort.Slice(dedupedDocs, func(i, j int) bool {
			return getScore(dedupedDocs[i]) > getScore(dedupedDocs[j])
		})

		// 6. 构建上下文（限制长度和 chunk 数）
		contextBuilder := strings.Builder{}
		totalLen := 0
		chunkCount := 0
		maxContextLen := 4000
		maxContextChunks := 10

		for _, doc := range dedupedDocs {
			content := getContent(doc) + "\n"

			// 检查 chunk 数限制
			if chunkCount >= maxContextChunks {
				break
			}

			// 检查长度限制
			if totalLen+len(content) > maxContextLen {
				break
			}

			contextBuilder.WriteString(content)
			totalLen += len(content)
			chunkCount++
		}

		state.RetrievedContext = contextBuilder.String()
		utils.DebugLog("上下文总长度: %d 字符, chunk 数: %d", totalLen, chunkCount)

		return state, nil
	}
}

// ContextBuilderNode 上下文构建节点
func ContextBuilderNode() func(ctx context.Context, state *RAGGraphState) (*RAGGraphState, error) {
	return func(ctx context.Context, state *RAGGraphState) (*RAGGraphState, error) {
		// 构建RAG prompt
		promptWord := fmt.Sprintf(`你是一个基于知识库的问答助手，请严格根据上下文回答。

【重要要求】
1. 必须完整列出上下文中的全部关键信息
2. 不要遗漏任何相关信息
3. 如果上下文中有多个相关点，请全部列出
4. 如果上下文中包含多个答案，必须完整列出全部答案，不要遗漏
5. 严格基于上下文回答，不要编造信息
6. 当前已提供RAG检索的上下文，请优先使用这些上下文回答。只有在上下文信息明显不足或用户明确要求补充检索时，才考虑调用search_docs工具进行补充检索。

【上下文】
%s

【问题】
%s

如果上下文中没有答案，请明确说不知道。`, state.RetrievedContext, state.UserInput)

		// 添加UserMessage到Messages
		state.Messages = append(state.Messages, schema.UserMessage(promptWord))

		return state, nil
	}
}

// ChatModelNode ChatModel节点
func ChatModelNode(chatModel model.BaseChatModel, toolList []tool.InvokableTool) func(ctx context.Context, state *RAGGraphState) (*RAGGraphState, error) {
	return func(ctx context.Context, state *RAGGraphState) (*RAGGraphState, error) {
		// 检查是否需要绑定工具
		var actualModel model.BaseChatModel = chatModel
		if len(toolList) > 0 {
			if tcModel, ok := chatModel.(interface {
				WithTools([]*schema.ToolInfo) (interface{}, error)
			}); ok {
				// 获取工具信息
				var toolInfos []*schema.ToolInfo
				for _, t := range toolList {
					info, err := t.Info(ctx)
					if err != nil {
						utils.DebugLog("获取工具信息失败: %v", err)
						continue
					}
					toolInfos = append(toolInfos, info)
				}

				if len(toolInfos) > 0 {
					modelWithTools, err := tcModel.WithTools(toolInfos)
					if err != nil {
						utils.DebugLog("绑定工具失败: %v", err)
					} else {
						actualModel = modelWithTools.(model.BaseChatModel)
					}
				}
			}
		}

		// 调用ChatModel
		reader, err := actualModel.Stream(ctx, state.Messages)
		if err != nil {
			return nil, fmt.Errorf("创建流式请求失败: %w", err)
		}
		defer reader.Close()

		// 处理流式内容
		var fullMessage *schema.Message
		for {
			chunk, err := reader.Recv()
			if err != nil {
				if errors.Is(err, io.EOF) {
					break
				}
				return nil, fmt.Errorf("接收流式数据失败: %w", err)
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

		state.ModelMessage = fullMessage

		// 如果没有tool calls，将内容作为最终答案
		if len(fullMessage.ToolCalls) == 0 {
			state.FinalAnswer = fullMessage.Content
		}

		return state, nil
	}
}

// ToolExecutionNode 工具执行节点
func ToolExecutionNode(toolList []tool.InvokableTool) func(ctx context.Context, state *RAGGraphState) (*RAGGraphState, error) {
	return func(ctx context.Context, state *RAGGraphState) (*RAGGraphState, error) {
		utils.DebugLog("检测到工具调用，数量: %d", len(state.ModelMessage.ToolCalls))

		// 执行工具调用（单轮）
		toolMessages := make([]*schema.Message, 0, len(state.ModelMessage.ToolCalls))
		for _, toolCall := range state.ModelMessage.ToolCalls {
			if toolCall.Function.Name != "" {
				utils.DebugLog("调用工具: id=%s, name=%s, 参数长度: %d", toolCall.ID, toolCall.Function.Name, len(toolCall.Function.Arguments))
				result, err := tools.ExecuteTool(ctx, toolList, toolCall.Function.Name, toolCall.Function.Arguments)
				if err != nil {
					utils.DebugLog("工具执行失败: %v", err)
					result = fmt.Sprintf("工具%s执行失败，请检查输入或重试", toolCall.Function.Name)
				}
				utils.DebugLog("工具执行成功，结果长度: %d", len(result))

				// 构造tool message
				toolMessage := schema.ToolMessage(toolCall.ID, result)
				toolMessages = append(toolMessages, toolMessage)
			}
		}

		state.ToolMessages = toolMessages
		return state, nil
	}
}

// FinalGenerationNode 最终生成节点
func FinalGenerationNode(chatModel model.BaseChatModel) func(ctx context.Context, state *RAGGraphState) (*RAGGraphState, error) {
	return func(ctx context.Context, state *RAGGraphState) (*RAGGraphState, error) {
		utils.DebugLog("基于工具结果生成最终答案")

		// 构建完整消息历史：Messages → ModelMessage → ToolMessages
		allMessages := make([]*schema.Message, 0, len(state.Messages)+1+len(state.ToolMessages))
		allMessages = append(allMessages, state.Messages...)
		allMessages = append(allMessages, state.ModelMessage)
		allMessages = append(allMessages, state.ToolMessages...)

		// 调用ChatModel生成最终答案
		reader, err := chatModel.Stream(ctx, allMessages)
		if err != nil {
			return nil, fmt.Errorf("生成最终答案失败: %w", err)
		}
		defer reader.Close()

		// 处理流式输出
		var fullMessage *schema.Message
		for {
			chunk, err := reader.Recv()
			if err != nil {
				if errors.Is(err, io.EOF) {
					break
				}
				return nil, fmt.Errorf("接收最终答案失败: %w", err)
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

		state.FinalAnswer = fullMessage.Content
		return state, nil
	}
}

// 辅助函数
func getScore(doc *schema.Document) float64 {
	if doc.MetaData == nil {
		return 0
	}
	if val, ok := doc.MetaData["score"].(float64); ok {
		return val
	}
	return 0
}

func getDocID(doc *schema.Document) string {
	if doc.MetaData == nil {
		return ""
	}
	if val, ok := doc.MetaData["doc_id"].(string); ok {
		return val
	}
	return ""
}

func getChunkIndex(doc *schema.Document) int64 {
	if doc.MetaData == nil {
		return 0
	}
	val, ok := doc.MetaData["chunk_index"]
	if !ok {
		return 0
	}
	if v, ok := val.(int64); ok {
		return v
	}
	if v, ok := val.(int); ok {
		return int64(v)
	}
	if v, ok := val.(float64); ok {
		return int64(v)
	}
	return 0
}

func getContent(doc *schema.Document) string {
	return doc.Content
}

func getDocDedupKey(doc *schema.Document) string {
	return fmt.Sprintf("%s:%d", getDocID(doc), getChunkIndex(doc))
}
