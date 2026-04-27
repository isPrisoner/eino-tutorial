package composition

import (
	"context"
	"fmt"

	"eino-tutorial/internal/retrieval"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
)

// RAGGraphState RAG Graph状态
type RAGGraphState struct {
	UserInput        string
	RetrievedContext string
	Messages         []*schema.Message // Graph内部使用的消息副本
	ModelMessage     *schema.Message
	ToolMessages     []*schema.Message
	FinalAnswer      string
}

// NewRAGGraph 创建RAG Graph并Compile
func NewRAGGraph(retrievalService *retrieval.Service, chatModel model.BaseChatModel, toolList []tool.InvokableTool) (compose.Runnable[*RAGGraphState, *RAGGraphState], error) {
	// 创建Graph：输入和输出都是RAGGraphState
	graph := compose.NewGraph[*RAGGraphState, *RAGGraphState]()

	// 1. RetrievalNode
	retrievalLambda := compose.InvokableLambda(RetrievalNode(retrievalService))
	if err := graph.AddLambdaNode("retrieval", retrievalLambda); err != nil {
		return nil, fmt.Errorf("添加retrieval节点失败: %w", err)
	}

	// 2. ContextBuilderNode
	contextBuilderLambda := compose.InvokableLambda(ContextBuilderNode())
	if err := graph.AddLambdaNode("context_builder", contextBuilderLambda); err != nil {
		return nil, fmt.Errorf("添加context_builder节点失败: %w", err)
	}

	// 3. ChatModelNode
	chatModelLambda := compose.InvokableLambda(ChatModelNode(chatModel, toolList))
	if err := graph.AddLambdaNode("chat_model", chatModelLambda); err != nil {
		return nil, fmt.Errorf("添加chat_model节点失败: %w", err)
	}

	// 4. ToolExecutionNode
	toolExecutionLambda := compose.InvokableLambda(ToolExecutionNode(toolList))
	if err := graph.AddLambdaNode("tool_execution", toolExecutionLambda); err != nil {
		return nil, fmt.Errorf("添加tool_execution节点失败: %w", err)
	}

	// 5. FinalGenerationNode
	finalGenerationLambda := compose.InvokableLambda(FinalGenerationNode(chatModel))
	if err := graph.AddLambdaNode("final_generation", finalGenerationLambda); err != nil {
		return nil, fmt.Errorf("添加final_generation节点失败: %w", err)
	}

	// 添加边
	if err := graph.AddEdge("retrieval", "context_builder"); err != nil {
		return nil, fmt.Errorf("添加边retrieval->context_builder失败: %w", err)
	}
	if err := graph.AddEdge("context_builder", "chat_model"); err != nil {
		return nil, fmt.Errorf("添加边context_builder->chat_model失败: %w", err)
	}
	if err := graph.AddEdge("tool_execution", "final_generation"); err != nil {
		return nil, fmt.Errorf("添加边tool_execution->final_generation失败: %w", err)
	}

	// 条件分支
	branch := compose.NewGraphBranch(
		func(ctx context.Context, state *RAGGraphState) (string, error) {
			if state.ModelMessage != nil && len(state.ModelMessage.ToolCalls) > 0 {
				return "tool_execution", nil
			}
			return compose.END, nil
		},
		map[string]bool{
			"tool_execution": true,
			compose.END:      true,
		},
	)
	if err := graph.AddBranch("chat_model", branch); err != nil {
		return nil, fmt.Errorf("添加分支chat_model失败: %w", err)
	}

	// Compile Graph
	runnable, err := graph.Compile(context.Background())
	if err != nil {
		return nil, fmt.Errorf("Graph编译失败: %w", err)
	}

	return runnable, nil
}
