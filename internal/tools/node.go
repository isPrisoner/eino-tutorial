package tools

import (
	"context"
	"fmt"

	"eino-tutorial/internal/retrieval"

	"github.com/cloudwego/eino/components/tool"
)

// NewToolsNode 创建工具列表
func NewToolsNode(ctx context.Context, retrievalService *retrieval.Service) ([]tool.InvokableTool, error) {
	// 创建工具注册中心
	registry := NewRegistry(retrievalService)

	// 获取工具实例
	toolInstances := registry.GetToolInstances()

	return toolInstances, nil
}

// ExecuteTool 手动执行工具
func ExecuteTool(ctx context.Context, tools []tool.InvokableTool, toolName string, argumentsJSON string) (string, error) {
	for _, t := range tools {
		info, err := t.Info(ctx)
		if err != nil {
			return "", err
		}
		if info.Name == toolName {
			return t.InvokableRun(ctx, argumentsJSON)
		}
	}
	return "", fmt.Errorf("tool not found: %s", toolName)
}
