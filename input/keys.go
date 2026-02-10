package input

import (
	tea "github.com/charmbracelet/bubbletea"
)

// HandleKeyPress processes keyboard input and returns appropriate actions
func HandleKeyPress(msg tea.KeyMsg) (action string, value string) {
	switch msg.Type {
	case tea.KeyCtrlC:
		return "quit", ""
	case tea.KeyUp:
		return "move", "up"
	case tea.KeyDown:
		return "move", "down"
	case tea.KeyEnter:
		return "select", ""
	}

	// Handle specific key presses
	switch msg.String() {
	case "q", "Q":
		return "quit", ""
	case "1":
		return "select_key", "1"
	case "2":
		return "select_key", "2"
	}

	return "", ""
}
