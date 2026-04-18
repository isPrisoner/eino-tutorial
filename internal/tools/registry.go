package tools

import (
	"context"
	"fmt"

	"eino-tutorial/internal/retrieval"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"
)

// Registry 工具注册中心
type Registry struct {
	calculator tool.InvokableTool
	searchDocs tool.InvokableTool
}

// NewRegistry 创建工具注册中心
func NewRegistry(retrievalService *retrieval.Service) *Registry {
	return &Registry{
		calculator: NewCalculator(),
		searchDocs: NewSearchDocs(retrievalService),
	}
}

// GetToolInfos 返回所有工具的信息列表
func (r *Registry) GetToolInfos(ctx context.Context) ([]*schema.ToolInfo, error) {
	var toolInfos []*schema.ToolInfo

	if info, err := r.calculator.Info(ctx); err != nil {
		// 跳过失败的工具，不中断整个流程
	} else {
		toolInfos = append(toolInfos, info)
	}

	if info, err := r.searchDocs.Info(ctx); err != nil {
		// 跳过失败的工具，不中断整个流程
	} else {
		toolInfos = append(toolInfos, info)
	}

	if len(toolInfos) == 0 {
		return nil, fmt.Errorf("没有可用的工具")
	}

	return toolInfos, nil
}

// GetToolInstances 返回所有工具的实例列表
func (r *Registry) GetToolInstances() []tool.InvokableTool {
	return []tool.InvokableTool{
		r.calculator,
		r.searchDocs,
	}
}
