package utils

// float64ToFloat32 将 float64 切片转换为 float32 切片
func float64ToFloat32(vec []float64) []float32 {
	if vec == nil {
		return nil
	}
	result := make([]float32, len(vec))
	for i, v := range vec {
		result[i] = float32(v)
	}
	return result
}

// Float64ToFloat32 将 float64 切片转换为 float32 切片（导出版本）
func Float64ToFloat32(vec []float64) []float32 {
	return float64ToFloat32(vec)
}

// float32ToFloat64 将 float32 切片转换为 float64 切片
func float32ToFloat64(vec []float32) []float64 {
	if vec == nil {
		return nil
	}
	result := make([]float64, len(vec))
	for i, v := range vec {
		result[i] = float64(v)
	}
	return result
}

// Float32ToFloat64 将 float32 切片转换为 float64 切片（导出版本）
func Float32ToFloat64(vec []float32) []float64 {
	return float32ToFloat64(vec)
}
