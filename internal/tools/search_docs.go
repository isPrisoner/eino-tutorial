package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"eino-tutorial/internal/retrieval"
	"eino-tutorial/internal/utils"

	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"
	"github.com/eino-contrib/jsonschema"
	orderedmap "github.com/wk8/go-ordered-map/v2"
)

// SearchDocs 文档搜索工具
type SearchDocs struct {
	retrievalService *retrieval.Service
}

// searchDocsInput 文档搜索输入参数结构
type searchDocsInput struct {
	Query string `json:"query"`
}

// Info 返回工具信息
func (s *SearchDocs) Info(ctx context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{
		Name: "search_docs",
		Desc: "在知识库中进行补充文档检索。重要：RAG流程已提供基础检索上下文，请优先使用现有上下文回答。仅在以下情况使用此工具：1) 上下文信息明显不足无法回答；2) 用户明确要求补充检索；3) 需要更全面的检索结果。避免不必要的重复检索。",
		ParamsOneOf: schema.NewParamsOneOfByJSONSchema(&jsonschema.Schema{
			Type: "object",
			Properties: orderedmap.New[string, *jsonschema.Schema](
				orderedmap.WithInitialData(
					orderedmap.Pair[string, *jsonschema.Schema]{
						Key: "query",
						Value: &jsonschema.Schema{
							Type:        "string",
							Description: "搜索查询语句",
						},
					},
				),
			),
			Required: []string{"query"},
		}),
	}, nil
}

// InvokableRun 执行文档搜索
func (s *SearchDocs) InvokableRun(ctx context.Context, argumentsInJSON string, opts ...tool.Option) (string, error) {
	// 解析JSON参数
	var input searchDocsInput
	if err := json.Unmarshal([]byte(argumentsInJSON), &input); err != nil {
		utils.DebugLog("search_docs工具解析参数失败: %v", err)
		return "", fmt.Errorf("解析参数失败: %w", err)
	}

	utils.DebugLog("search_docs工具调用，查询: %s", input.Query)

	// 调用retrieval service进行检索
	docs, err := s.retrievalService.Retrieve(ctx, input.Query, retriever.WithTopK(5))
	if err != nil {
		utils.DebugLog("search_docs工具检索失败: %v", err)
		return "", fmt.Errorf("文档检索失败: %w", err)
	}

	utils.DebugLog("search_docs工具检索完成，找到文档数: %d", len(docs))

	// 格式化检索结果
	if len(docs) == 0 {
		utils.DebugLog("search_docs工具未找到相关文档")
		return "未找到相关文档", nil
	}

	var result strings.Builder
	result.WriteString(fmt.Sprintf("找到 %d 个相关文档：\n", len(docs)))
	for i, doc := range docs {
		result.WriteString(fmt.Sprintf("\n[文档%d]\n", i+1))
		result.WriteString(fmt.Sprintf("内容: %s\n", doc.Content))
		if doc.MetaData != nil {
			if docID, ok := doc.MetaData["doc_id"].(string); ok {
				result.WriteString(fmt.Sprintf("文档ID: %s\n", docID))
			}
		}
	}

	utils.DebugLog("search_docs工具执行成功，结果长度: %d", result.Len())
	return result.String(), nil
}

// NewSearchDocs 创建文档搜索工具实例
func NewSearchDocs(retrievalService *retrieval.Service) tool.InvokableTool {
	return &SearchDocs{
		retrievalService: retrievalService,
	}
}
