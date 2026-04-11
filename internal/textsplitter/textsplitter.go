package textsplitter

import (
	"strings"
)

// TextSplitter 文本切分器
type TextSplitter struct {
	ChunkSize    int // 每个分块的最大字符数
	ChunkOverlap int // 分块之间的重叠字符数
}

// NewTextSplitter 创建文本切分器
func NewTextSplitter(chunkSize, chunkOverlap int) *TextSplitter {
	return &TextSplitter{
		ChunkSize:    chunkSize,
		ChunkOverlap: chunkOverlap,
	}
}

// Split 切分文本
func (ts *TextSplitter) Split(text string) []string {
	// 1. trim 空白
	text = strings.TrimSpace(text)
	if text == "" {
		return []string{}
	}

	// 2. 按字符数切分
	var chunks []string
	runes := []rune(text)
	length := len(runes)

	for i := 0; i < length; i += ts.ChunkSize - ts.ChunkOverlap {
		end := i + ts.ChunkSize
		if end > length {
			end = length
		}

		chunk := string(runes[i:end])
		chunk = strings.TrimSpace(chunk)

		// 3. 过滤空 chunk
		if chunk == "" {
			continue
		}

		// 4. 过滤过短 chunk（小于 10 字符）
		if len([]rune(chunk)) < 10 {
			continue
		}

		chunks = append(chunks, chunk)

		// 如果已经到了末尾，退出
		if end >= length {
			break
		}
	}

	return chunks
}

// SplitWithDocID 切分文本并生成文档 ID
func (ts *TextSplitter) SplitWithDocID(docID, text string) []string {
	chunks := ts.Split(text)
	return chunks
}

// CountChunks 计算切分后的分块数量
func (ts *TextSplitter) CountChunks(text string) int {
	return len(ts.Split(text))
}
