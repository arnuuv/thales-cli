package animation

import (
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

type TickMsg time.Time

func AnimationTicker() tea.Cmd {
	return tea.Tick(time.Second/30, func(t time.Time) tea.Msg {
		return TickMsg(t)
	})
}
