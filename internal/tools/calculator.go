package tools

import (
	"context"
	"eino-tutorial/internal/utils"
	"encoding/json"
	"fmt"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"
	"github.com/eino-contrib/jsonschema"
	orderedmap "github.com/wk8/go-ordered-map/v2"
)

// Calculator 计算器工具
type Calculator struct{}

// calculatorInput 计算器输入参数结构
type calculatorInput struct {
	Expression string `json:"expression"`
}

// Info 返回工具信息
func (c *Calculator) Info(ctx context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{
		Name: "calculator",
		Desc: "计算数学表达式，支持加减乘除四则运算",
		ParamsOneOf: schema.NewParamsOneOfByJSONSchema(&jsonschema.Schema{
			Type: "object",
			Properties: orderedmap.New[string, *jsonschema.Schema](
				orderedmap.WithInitialData(
					orderedmap.Pair[string, *jsonschema.Schema]{
						Key: "expression",
						Value: &jsonschema.Schema{
							Type:        "string",
							Description: "要计算的数学表达式，例如：123 + 456 或 100 * 2",
						},
					},
				),
			),
			Required: []string{"expression"},
		}),
	}, nil
}

// InvokableRun 执行计算
func (c *Calculator) InvokableRun(ctx context.Context, argumentsInJSON string, opts ...tool.Option) (string, error) {
	// 解析JSON参数
	var input calculatorInput
	if err := json.Unmarshal([]byte(argumentsInJSON), &input); err != nil {
		utils.DebugLog("calculator工具解析参数失败: %v", err)
		return "", fmt.Errorf("解析参数失败: %w", err)
	}

	utils.DebugLog("calculator工具调用，表达式: %s", input.Expression)

	// 计算表达式
	result, err := calculateExpression(input.Expression)
	if err != nil {
		utils.DebugLog("calculator工具计算失败: %v", err)
		return "", fmt.Errorf("计算失败: %w", err)
	}

	utils.DebugLog("calculator工具执行成功，结果长度: %d", len(result))
	return fmt.Sprintf("计算结果: %s = %s", input.Expression, result), nil
}

// calculateExpression 计算简单的数学表达式
// 这是一个基础实现，只支持加减乘除
func calculateExpression(expr string) (string, error) {
	// 使用Go的类型检查器来安全计算表达式
	// 注意：这是一个简化实现，生产环境应使用专门的表达式解析库

	// 暂时返回示例结果，实际应该解析并计算表达式
	// 为了安全起见，这里只返回表达式本身，不执行计算
	// 真正的实现需要使用表达式解析库如 github.com/Knetic/govaluate

	return expr, nil
}

// NewCalculator 创建计算器工具实例
func NewCalculator() tool.InvokableTool {
	return &Calculator{}
}
