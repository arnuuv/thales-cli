package runner

import (
	"archive/zip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

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

// TerrainGenDoneMsg is sent when full terrain generation (all resolutions) finishes
type TerrainGenDoneMsg struct {
	OutDir string
	Err    error
}

// MaritimeGenDoneMsg is sent when full maritime generation finishes
type MaritimeGenDoneMsg struct {
	OutDir string
	Err    error
}

// DatasetZipDoneMsg is sent when dataset ZIP packaging finishes
type DatasetZipDoneMsg struct {
	ZipPath string
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

// RunTerrainGeneration runs terrain generation for all selected resolutions and area size
func RunTerrainGeneration(baseDir string, resolutions []string, areaKm float64) tea.Cmd {
	return func() tea.Msg {
		// Create timestamp for this run
		timestamp := time.Now().Format("2006_01_02_150405")
		runDir := filepath.Join(baseDir, "output", "terrain", fmt.Sprintf("run_%s", timestamp))
		
		if err := os.MkdirAll(runDir, 0755); err != nil {
			return TerrainGenDoneMsg{Err: fmt.Errorf("failed to create output directory: %w", err)}
		}

		script := filepath.Join(baseDir, "generate_dataset.py")
		if _, err := os.Stat(script); err != nil {
			return TerrainGenDoneMsg{Err: fmt.Errorf("generate_dataset.py not found: %w", err)}
		}

		// Calculate bounds based on area size
		// For simplicity, create a square area centered on default location
		centerLon := (DefaultMinLon + DefaultMaxLon) / 2
		centerLat := (DefaultMinLat + DefaultMaxLat) / 2
		
		// Rough conversion: 1 degree ≈ 111 km at equator
		// For latitude, this is fairly accurate
		// For longitude, it varies with latitude, but we'll use a simple approximation
		latDelta := areaKm / 111.0 / 2
		lonDelta := areaKm / (111.0 * 0.8) / 2 // Approximate for mid-latitudes
		
		minLon := centerLon - lonDelta
		maxLon := centerLon + lonDelta
		minLat := centerLat - latDelta
		maxLat := centerLat + latDelta

		// Generate for each resolution
		for _, res := range resolutions {
			fmt.Printf("\n🗺️  Generating terrain at %s resolution...\n", res)
			
			resDir := filepath.Join(runDir, res)
			if err := os.MkdirAll(resDir, 0755); err != nil {
				return TerrainGenDoneMsg{Err: fmt.Errorf("failed to create resolution directory: %w", err)}
			}

			// Format bounds as separate arguments
			cmd := exec.Command("python3", script,
				"--resolutions", res,
				"--bounds", 
				fmt.Sprintf("%.6f", minLon),
				fmt.Sprintf("%.6f", minLat),
				fmt.Sprintf("%.6f", maxLon),
				fmt.Sprintf("%.6f", maxLat),
				"--outdir", resDir,
				"--biome", "city",
				"--seed", "42",
			)
			cmd.Dir = baseDir
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			
			if err := cmd.Run(); err != nil {
				return TerrainGenDoneMsg{Err: fmt.Errorf("terrain generation failed for %s: %w", res, err)}
			}
			
			fmt.Printf("✅ Completed %s resolution\n", res)
		}

		// Generate metadata.json
		metadata := map[string]interface{}{
			"generator":   "Thales Synthetic EO Generator",
			"timestamp":   timestamp,
			"area_km":     areaKm,
			"resolutions": resolutions,
			"bounds": map[string]float64{
				"min_lon": minLon,
				"min_lat": minLat,
				"max_lon": maxLon,
				"max_lat": maxLat,
			},
		}

		metadataPath := filepath.Join(runDir, "metadata.json")
		metadataJSON, err := json.MarshalIndent(metadata, "", "  ")
		if err != nil {
			return TerrainGenDoneMsg{Err: fmt.Errorf("failed to create metadata: %w", err)}
		}

		if err := os.WriteFile(metadataPath, metadataJSON, 0644); err != nil {
			return TerrainGenDoneMsg{Err: fmt.Errorf("failed to write metadata: %w", err)}
		}

		return TerrainGenDoneMsg{OutDir: runDir, Err: nil}
	}
}

// RunMaritimeGeneration runs full maritime generation producing all GeoJSON files
func RunMaritimeGeneration(baseDir string) tea.Cmd {
	return func() tea.Msg {
		// Create timestamp for this run
		timestamp := time.Now().Format("2006_01_02_150405")
		runDir := filepath.Join(baseDir, "output", "maritime", fmt.Sprintf("run_%s", timestamp))
		
		if err := os.MkdirAll(runDir, 0755); err != nil {
			return MaritimeGenDoneMsg{Err: fmt.Errorf("failed to create output directory: %w", err)}
		}

	// Use the comprehensive maritime generator script
	script := filepath.Join(baseDir, "maritime_generator", "generate_all_outputs.py")
	if _, err := os.Stat(script); err != nil {
		return MaritimeGenDoneMsg{Err: fmt.Errorf("generate_all_outputs.py not found: %w", err)}
	}

	fmt.Println("\n🌊 Generating maritime data...")

	// Generate all maritime outputs using the Python script
	cmd := exec.Command("python3", script,
		"--output-dir", runDir,
		"--num-vessels", "10",
		"--num-lanes", "4",
		"--num-assets", "15",
		"--num-routes", "10",
		"--seed", "42",
	)
	cmd.Dir = baseDir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	
	if err := cmd.Run(); err != nil {
		return MaritimeGenDoneMsg{Err: fmt.Errorf("maritime generation failed: %w", err)}
	}

		// Generate metadata
		metadata := map[string]interface{}{
			"generator":  "Thales Maritime Generator",
			"timestamp":  timestamp,
			"outputs": []string{
				"vessels.geojson",
				"sea_lanes.geojson",
				"coastal_assets.geojson",
				"routes.geojson",
			},
		}

		metadataPath := filepath.Join(runDir, "metadata.json")
		metadataJSON, err := json.MarshalIndent(metadata, "", "  ")
		if err != nil {
			return MaritimeGenDoneMsg{Err: fmt.Errorf("failed to create metadata: %w", err)}
		}

		if err := os.WriteFile(metadataPath, metadataJSON, 0644); err != nil {
			return MaritimeGenDoneMsg{Err: fmt.Errorf("failed to write metadata: %w", err)}
		}

		fmt.Println("✅ Maritime generation complete")

		return MaritimeGenDoneMsg{OutDir: runDir, Err: nil}
	}
}


// CreateDatasetZip packages terrain and/or maritime outputs into a ZIP file
func CreateDatasetZip(baseDir string, terrainDir string, maritimeDir string) tea.Cmd {
	return func() tea.Msg {
		timestamp := time.Now().Format("2006_01_02_150405")
		downloadsDir := filepath.Join(baseDir, "output", "downloads")
		
		if err := os.MkdirAll(downloadsDir, 0755); err != nil {
			return DatasetZipDoneMsg{Err: fmt.Errorf("failed to create downloads directory: %w", err)}
		}

		zipPath := filepath.Join(downloadsDir, fmt.Sprintf("thales_dataset_%s.zip", timestamp))
		
		zipFile, err := os.Create(zipPath)
		if err != nil {
			return DatasetZipDoneMsg{Err: fmt.Errorf("failed to create zip file: %w", err)}
		}
		defer zipFile.Close()

		zipWriter := zip.NewWriter(zipFile)
		defer zipWriter.Close()

		// Add terrain directory if it exists
		if terrainDir != "" {
			if err := addDirToZip(zipWriter, terrainDir, "terrain"); err != nil {
				return DatasetZipDoneMsg{Err: fmt.Errorf("failed to add terrain to zip: %w", err)}
			}
		}

		// Add maritime directory if it exists
		if maritimeDir != "" {
			if err := addDirToZip(zipWriter, maritimeDir, "maritime"); err != nil {
				return DatasetZipDoneMsg{Err: fmt.Errorf("failed to add maritime to zip: %w", err)}
			}
		}

		// Create a top-level metadata file
		metadata := map[string]interface{}{
			"generator":    "Thales Synthetic Data Generator",
			"timestamp":    timestamp,
			"has_terrain":  terrainDir != "",
			"has_maritime": maritimeDir != "",
		}

		metadataJSON, err := json.MarshalIndent(metadata, "", "  ")
		if err != nil {
			return DatasetZipDoneMsg{Err: fmt.Errorf("failed to create metadata: %w", err)}
		}

		metadataWriter, err := zipWriter.Create("metadata.json")
		if err != nil {
			return DatasetZipDoneMsg{Err: fmt.Errorf("failed to add metadata to zip: %w", err)}
		}

		if _, err := metadataWriter.Write(metadataJSON); err != nil {
			return DatasetZipDoneMsg{Err: fmt.Errorf("failed to write metadata to zip: %w", err)}
		}

		fmt.Printf("\n📦 Dataset packaged: %s\n", zipPath)

		return DatasetZipDoneMsg{ZipPath: zipPath, Err: nil}
	}
}

// addDirToZip recursively adds a directory to a zip archive
func addDirToZip(zipWriter *zip.Writer, sourceDir string, basePath string) error {
	return filepath.Walk(sourceDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip directories
		if info.IsDir() {
			return nil
		}

		// Get relative path
		relPath, err := filepath.Rel(sourceDir, path)
		if err != nil {
			return err
		}

		// Create zip path
		zipPath := filepath.Join(basePath, relPath)
		// Normalize path separators for zip (always use forward slashes)
		zipPath = strings.ReplaceAll(zipPath, string(filepath.Separator), "/")

		// Create file in zip
		writer, err := zipWriter.Create(zipPath)
		if err != nil {
			return err
		}

		// Open source file
		file, err := os.Open(path)
		if err != nil {
			return err
		}
		defer file.Close()

		// Copy file contents
		_, err = io.Copy(writer, file)
		return err
	})
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
		cmd := exec.Command("python3", script, "--output", outPath)
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
