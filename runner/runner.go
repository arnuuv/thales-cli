package runner

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"

	tea "github.com/charmbracelet/bubbletea"
)

// Default bounds for map generation (small area for quick runs)
const (
	DefaultMinLon = -122.5
	DefaultMinLat = 37.5
	DefaultMaxLon = -122.3
	DefaultMaxLat = 37.7
)

// MapGenDoneMsg is sent when map generation finishes
type MapGenDoneMsg struct {
	OutDir string
	Err    error
}

// MaritimeGeoJSONDoneMsg is sent when maritime GeoJSON production finishes
type MaritimeGeoJSONDoneMsg struct {
	OutPath string
	Err     error
}

// RunMapGeneration runs generate_dataset.py for the given resolution (25cm, 1m, 10m).
// BaseDir is the repo root (where generate_dataset.py lives).
func RunMapGeneration(baseDir, resolution, outDir string) tea.Cmd {
	return func() tea.Msg {
		script := filepath.Join(baseDir, "generate_dataset.py")
		if _, err := os.Stat(script); err != nil {
			return MapGenDoneMsg{Err: fmt.Errorf("generate_dataset.py not found: %w", err)}
		}
		bounds := fmt.Sprintf("%.2f %.2f %.2f %.2f",
			DefaultMinLon, DefaultMinLat, DefaultMaxLon, DefaultMaxLat)
		cmd := exec.Command("python", script,
			"--resolutions", resolution,
			"--bounds", bounds,
			"--outdir", outDir,
			"--biome", "city",
			"--seed", "42",
		)
		cmd.Dir = baseDir
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		err := cmd.Run()
		return MapGenDoneMsg{OutDir: outDir, Err: err}
	}
}

// RunMaritimeGeoJSON runs maritime_generator/produce_geojson_routes.py and writes routes.geojson.
func RunMaritimeGeoJSON(baseDir, outPath string) tea.Cmd {
	return func() tea.Msg {
		script := filepath.Join(baseDir, "maritime_generator", "produce_geojson_routes.py")
		if _, err := os.Stat(script); err != nil {
			return MaritimeGeoJSONDoneMsg{Err: fmt.Errorf("produce_geojson_routes.py not found: %w", err)}
		}
		if err := os.MkdirAll(filepath.Dir(outPath), 0755); err != nil {
			return MaritimeGeoJSONDoneMsg{Err: err}
		}
		cmd := exec.Command("python", script, "--output", outPath)
		cmd.Dir = baseDir
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		err := cmd.Run()
		return MaritimeGeoJSONDoneMsg{OutPath: outPath, Err: err}
	}
}

// OpenOutputPath opens the file's parent folder in the OS file manager (for "download").
func OpenOutputPath(path string) tea.Cmd {
	return func() tea.Msg {
		abs, err := filepath.Abs(path)
		if err != nil {
			return nil
		}
		dir := filepath.Dir(abs)
		var cmd *exec.Cmd
		switch runtime.GOOS {
		case "windows":
			// Explorer: select the file (or open folder)
			cmd = exec.Command("explorer", "/select,"+abs)
		case "darwin":
			cmd = exec.Command("open", "-R", abs)
		default:
			cmd = exec.Command("xdg-open", dir)
		}
		_ = cmd.Start()
		return nil
	}
}
