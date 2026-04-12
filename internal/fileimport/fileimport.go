package fileimport

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// FileImportResult 文件导入结果
type FileImportResult struct {
	DocID      string // 相对路径，作为稳定标识
	ChunkCount int    // 分块数量
	Success    bool   // 是否成功
	Error      error  // 错误信息
}

// ReadFileContent 读取文件内容（UTF-8）
func ReadFileContent(filePath string) (string, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return "", fmt.Errorf("读取文件失败: %w", err)
	}
	return string(content), nil
}

// GetRelativePath 获取相对于工作目录的路径
func GetRelativePath(absPath string) (string, error) {
	wd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	relPath, err := filepath.Rel(wd, absPath)
	if err != nil {
		return "", err
	}
	return relPath, nil
}

// IsHiddenFile 判断是否为隐藏文件或目录
func IsHiddenFile(path string) bool {
	base := filepath.Base(path)
	return strings.HasPrefix(base, ".")
}

// ScanDirectory 递归扫描目录，返回所有 .txt 和 .md 文件
func ScanDirectory(dir string) ([]string, error) {
	var files []string

	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// 跳过隐藏文件和目录
		if IsHiddenFile(path) {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		// 只处理文件
		if !info.IsDir() {
			ext := strings.ToLower(filepath.Ext(path))
			if ext == ".txt" || ext == ".md" {
				files = append(files, path)
			}
		}

		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("扫描目录失败: %w", err)
	}

	return files, nil
}
