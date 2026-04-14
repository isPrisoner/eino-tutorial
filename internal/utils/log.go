package utils

import (
	"log"
	"os"
)

var (
	// DebugMode 仅在 DEBUG=true 时输出日志
	DebugMode = os.Getenv("DEBUG") == "true"
)

// DebugLog 仅在 DEBUG=true 时输出日志
func DebugLog(format string, args ...interface{}) {
	if DebugMode {
		log.Printf(format, args...)
	}
}
