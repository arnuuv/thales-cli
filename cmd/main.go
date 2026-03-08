package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/arnavanchan/thales-cli/animation"
	"github.com/arnavanchan/thales-cli/input"
	"github.com/arnavanchan/thales-cli/runner"
	"github.com/arnavanchan/thales-cli/state"
	"github.com/arnavanchan/thales-cli/ui"

	tea "github.com/charmbracelet/bubbletea"
)

type Model struct {
	appState       *state.AppState
	rotation       *animation.RotationState
	layout         *ui.Layout
	ready          bool
	showingStartup bool
	startupTime    time.Time
}

func InitialModel() Model {
	return Model{
		appState:       state.NewAppState(),
		rotation:       animation.NewRotationState(),
		layout:         ui.NewLayout(ui.GetTerminalSize()),
		ready:          false,
		showingStartup: true,
		startupTime:    time.Now(),
	}
}

func (m Model) Init() tea.Cmd {
	return tea.Batch(
		animation.AnimationTicker(),
		showStartup(),
	)
}

func showStartup() tea.Cmd {
	return tea.Tick(2*time.Second, func(t time.Time) tea.Msg {
		return startupDoneMsg{}
	})
}

type startupDoneMsg struct{}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {

	case tea.WindowSizeMsg:
		m.layout = ui.NewLayout(msg.Width, msg.Height)
		m.ready = true
		return m, nil

	case startupDoneMsg:
		m.showingStartup = false
		return m, nil

	case animation.TickMsg:
		m.rotation.Update(time.Time(msg))
		return m, animation.AnimationTicker()

	case tea.KeyMsg:
		action, value := input.HandleKeyPress(msg)

		switch action {
		case "quit":
			return m, tea.Quit

		case "move":
			if value == "up" {
				m.appState.MoveSelection(-1)
			} else if value == "down" {
				m.appState.MoveSelection(1)
			}

		case "select":
			currentOpt := m.appState.GetCurrentOption()
			return handleSelection(m, currentOpt.Key)

		case "select_key":
			return handleSelection(m, value)

		case "download":
			return m, openLastOutput(m)
		}

		return m, nil

	case runner.TerrainGenDoneMsg:
		m.appState.GeneratingMap = false
		m.appState.LastMapOutputDir = msg.OutDir
		if msg.Err != nil {
			m.appState.Status = "Terrain generation failed: " + msg.Err.Error()
		} else {
			m.appState.Status = "Terrain generation complete. Creating dataset ZIP..."
			// Automatically create ZIP after terrain generation
			return m, runner.CreateDatasetZip(getBaseDir(), msg.OutDir, "")
		}
		return m, nil

	case runner.MaritimeGenDoneMsg:
		m.appState.GeneratingMaritime = false
		m.appState.LastMaritimeGeoJSONPath = msg.OutDir
		if msg.Err != nil {
			m.appState.Status = "Maritime generation failed: " + msg.Err.Error()
		} else {
			m.appState.Status = "Maritime generation complete. Creating dataset ZIP..."
			// Automatically create ZIP after maritime generation
			return m, runner.CreateDatasetZip(getBaseDir(), "", msg.OutDir)
		}
		return m, nil

	case runner.DatasetZipDoneMsg:
		if msg.Err != nil {
			m.appState.Status = "ZIP packaging failed: " + msg.Err.Error()
		} else {
			m.appState.LastDatasetZipPath = msg.ZipPath
			m.appState.Status = "Dataset ready! Press [D] to open downloads folder."
		}
		return m, nil

	case runner.MapGenDoneMsg:
		m.appState.GeneratingMap = false
		m.appState.LastMapOutputDir = msg.OutDir
		if msg.Err != nil {
			m.appState.Status = "Map generation failed: " + msg.Err.Error()
		} else {
			m.appState.Status = "Map generation complete. Press [D] to open output folder."
		}
		return m, nil

	case runner.MaritimeGeoJSONDoneMsg:
		m.appState.GeneratingMaritime = false
		m.appState.LastMaritimeGeoJSONPath = msg.OutPath
		if msg.Err != nil {
			m.appState.Status = "Maritime GeoJSON failed: " + msg.Err.Error()
		} else {
			m.appState.Status = "GeoJSON routes ready. Press [D] to download/open."
		}
		return m, nil
	}

	return m, nil
}

func (m Model) View() string {
	if !m.ready {
		return "Loading..."
	}

	if m.showingStartup {
		return ui.RenderStartupTitle(m.layout.Width, m.layout.Height)
	}

	return m.layout.Render(m.appState, m.rotation)
}

// handleSelection processes user selection based on current state
func handleSelection(m Model, key string) (Model, tea.Cmd) {
	baseDir := getBaseDir()

	// Check if we're in a selection workflow first (priority over main menu)
	// Handle resolution selection
	if m.appState.TerrainSelectionStep == "resolution" {
		switch key {
		case "1":
			m.appState.ToggleResolution("1m")
			m.appState.Status = fmt.Sprintf("Selected: %v. Press [Enter] to continue or select more.", m.appState.SelectedResolutions)
			return m, nil
		case "2":
			m.appState.ToggleResolution("10m")
			m.appState.Status = fmt.Sprintf("Selected: %v. Press [Enter] to continue or select more.", m.appState.SelectedResolutions)
			return m, nil
		case "3":
			m.appState.ToggleResolution("25cm")
			m.appState.Status = fmt.Sprintf("Selected: %v. Press [Enter] to continue or select more.", m.appState.SelectedResolutions)
			return m, nil
		case "4":
			m.appState.SelectAllResolutions()
			m.appState.Status = fmt.Sprintf("Selected: All resolutions. Press [Enter] to continue.")
			return m, nil
		case "enter":
			if len(m.appState.SelectedResolutions) == 0 {
				m.appState.Status = "Error: Please select at least one resolution."
				return m, nil
			}
			m.appState.TerrainSelectionStep = "area"
			m.appState.Status = "Select area size: [1] 1km x 1km  [2] 2km x 2km  [Enter] Generate"
			return m, nil
		}
		return m, nil
	}

	// Handle area size selection
	if m.appState.TerrainSelectionStep == "area" {
		switch key {
		case "1":
			m.appState.SelectedAreaKm = 1.0
			m.appState.Status = "Selected: 1km x 1km. Press [Enter] to start generation."
			return m, nil
		case "2":
			m.appState.SelectedAreaKm = 2.0
			m.appState.Status = "Selected: 2km x 2km. Press [Enter] to start generation."
			return m, nil
		case "enter":
			if m.appState.SelectedAreaKm == 0 {
				m.appState.Status = "Error: Please select an area size."
				return m, nil
			}
			// Start terrain generation
			m.appState.TerrainSelectionStep = "done"
			m.appState.GeneratingMap = true
			m.appState.Status = fmt.Sprintf("Generating terrain: %v at %.0fkm x %.0fkm...", 
				m.appState.SelectedResolutions, m.appState.SelectedAreaKm, m.appState.SelectedAreaKm)
			return m, runner.RunTerrainGeneration(baseDir, m.appState.SelectedResolutions, m.appState.SelectedAreaKm)
		}
		return m, nil
	}

	// Handle main menu selections (only if not in a workflow)
	if m.appState.TerrainSelectionStep == "" && m.appState.MaritimeSelectionStep == "" {
		// Handle terrain selection workflow
		if key == "1" {
			// Start terrain selection workflow
			m.appState.ResetTerrainSelection()
			m.appState.TerrainSelectionStep = "resolution"
			m.appState.Status = "Select resolution(s): [1] 1m  [2] 10m  [3] 25cm  [4] All  [Enter] Continue"
			return m, nil
		}

		// Handle maritime selection
		if key == "2" {
			// Start maritime generation
			m.appState.GeneratingMaritime = true
			m.appState.Status = "Generating maritime data (vessels, sea lanes, coastal assets, routes)..."
			return m, runner.RunMaritimeGeneration(baseDir)
		}
	}

	// Legacy single-resolution map generation (for backward compatibility)
	if state.IsMapResolutionOption(key) {
		res := state.ResolutionForMapOption(key)
		outDir := filepath.Join(baseDir, "output", "maps")
		m.appState.GeneratingMap = true
		m.appState.Status = "Generating map at " + res + "..."
		return m, runner.RunMapGeneration(baseDir, res, outDir)
	}

	// Legacy maritime GeoJSON option
	if state.IsMaritimeGeoJSONOption(key) {
		outPath := filepath.Join(baseDir, "output", "maritime", "routes.geojson")
		m.appState.GeneratingMaritime = true
		m.appState.Status = "Producing GeoJSON routes..."
		return m, runner.RunMaritimeGeoJSON(baseDir, outPath)
	}

	return m, nil
}

// openLastOutput opens the last generated output (ZIP file, map folder, or maritime GeoJSON file).
func openLastOutput(m Model) tea.Cmd {
	if m.appState.LastDatasetZipPath != "" {
		return runner.OpenOutputPath(m.appState.LastDatasetZipPath)
	}
	if m.appState.LastMaritimeGeoJSONPath != "" {
		return runner.OpenOutputPath(m.appState.LastMaritimeGeoJSONPath)
	}
	if m.appState.LastMapOutputDir != "" {
		return runner.OpenOutputPath(m.appState.LastMapOutputDir)
	}
	return nil
}

// getBaseDir returns the base directory (repo root)
func getBaseDir() string {
	baseDir, err := os.Getwd()
	if err != nil {
		baseDir = "."
	}
	return baseDir
}

func main() {
	p := tea.NewProgram(
		InitialModel(),
		tea.WithAltScreen(),
		tea.WithMouseCellMotion(),
	)

	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error running program: %v\n", err)
		os.Exit(1)
	}
}
