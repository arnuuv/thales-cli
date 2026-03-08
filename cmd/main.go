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
	appState      *state.AppState
	rotation      *animation.RotationState
	layout        *ui.Layout
	ready         bool
	showingStartup bool
	startupTime   time.Time
}

func InitialModel() Model {
	return Model{
		appState:      state.NewAppState(),
		rotation:      animation.NewRotationState(),
		layout:        ui.NewLayout(ui.GetTerminalSize()),
		ready:         false,
		showingStartup: true,
		startupTime:   time.Now(),
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
			m.appState.SelectOption(currentOpt.Key)
			// Run generation if applicable
			m, cmd := runGenerationIfSelected(m, currentOpt.Key)
			return m, cmd

		case "select_key":
			m.appState.SelectOption(value)
			m, cmd := runGenerationIfSelected(m, value)
			return m, cmd

		case "download":
			return m, openLastOutput(m)
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

// runGenerationIfSelected starts map or maritime generation if the selected option is 1a/1b/1c or 2d.
func runGenerationIfSelected(m Model, key string) (Model, tea.Cmd) {
	baseDir, err := os.Getwd()
	if err != nil {
		baseDir = "."
	}
	if state.IsMapResolutionOption(key) {
		res := state.ResolutionForMapOption(key)
		outDir := filepath.Join(baseDir, "output", "maps")
		m.appState.GeneratingMap = true
		m.appState.Status = "Generating map at " + res + "..."
		return m, runner.RunMapGeneration(baseDir, res, outDir)
	}
	if state.IsMaritimeGeoJSONOption(key) {
		outPath := filepath.Join(baseDir, "output", "maritime", "routes.geojson")
		m.appState.GeneratingMaritime = true
		m.appState.Status = "Producing GeoJSON routes..."
		return m, runner.RunMaritimeGeoJSON(baseDir, outPath)
	}
	return m, nil
}

// openLastOutput opens the last generated output (map folder or maritime GeoJSON file).
func openLastOutput(m Model) tea.Cmd {
	if m.appState.LastMaritimeGeoJSONPath != "" {
		return runner.OpenOutputPath(m.appState.LastMaritimeGeoJSONPath)
	}
	if m.appState.LastMapOutputDir != "" {
		return runner.OpenOutputPath(m.appState.LastMapOutputDir)
	}
	return nil
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
