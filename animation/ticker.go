package animation

import (
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// TickMsg is sent on every animation frame
type TickMsg time.Time

// AnimationTicker creates a ticker for smooth animation
func AnimationTicker() tea.Cmd {
	return tea.Tick(time.Second/30, func(t time.Time) tea.Msg {
		return TickMsg(t)
	})
}
