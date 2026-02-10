package main

import (
	"fmt"
	"os"
	"time"

	"github.com/arnavanchan/thales-cli/animation"
	"github.com/arnavanchan/thales-cli/input"
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

		case "select_key":
			m.appState.SelectOption(value)
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
