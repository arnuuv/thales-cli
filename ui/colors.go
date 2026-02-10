package ui

import (
	"fmt"
	"math"

	"github.com/charmbracelet/lipgloss"
)

// RGB color definitions for the neon theme
var (
	CyanRGB     = lipgloss.Color("#00FFFF")
	TealRGB     = lipgloss.Color("#00CED1")
	PurpleRGB   = lipgloss.Color("#9370DB")
	ElectricRGB = lipgloss.Color("#7DF9FF")
	GreenRGB    = lipgloss.Color("#00FF9F")
	DimGray     = lipgloss.Color("#4A5568")
)

// RGB represents a 24-bit color
type RGB struct {
	R, G, B uint8
}

// Predefined RGB colors
var (
	Cyan     = RGB{0, 255, 255}
	Teal     = RGB{0, 206, 209}
	Purple   = RGB{147, 112, 219}
	Electric = RGB{125, 249, 255}
	Green    = RGB{0, 255, 159}
	Gray     = RGB{74, 85, 104}
)

// ToANSI converts RGB to ANSI escape sequence
func (c RGB) ToANSI() string {
	return fmt.Sprintf("\x1b[38;2;%d;%d;%dm", c.R, c.G, c.B)
}

// ToANSIBG converts RGB to ANSI background escape sequence
func (c RGB) ToANSIBG() string {
	return fmt.Sprintf("\x1b[48;2;%d;%d;%dm", c.R, c.G, c.B)
}

// Dim reduces the brightness of a color
func (c RGB) Dim(factor float64) RGB {
	return RGB{
		R: uint8(float64(c.R) * factor),
		G: uint8(float64(c.G) * factor),
		B: uint8(float64(c.B) * factor),
	}
}

// Brighten increases the brightness of a color
func (c RGB) Brighten(factor float64) RGB {
	return RGB{
		R: uint8(math.Min(float64(c.R)*factor, 255)),
		G: uint8(math.Min(float64(c.G)*factor, 255)),
		B: uint8(math.Min(float64(c.B)*factor, 255)),
	}
}

// Lerp linearly interpolates between two colors
func Lerp(c1, c2 RGB, t float64) RGB {
	return RGB{
		R: uint8(float64(c1.R) + (float64(c2.R)-float64(c1.R))*t),
		G: uint8(float64(c1.G) + (float64(c2.G)-float64(c1.G))*t),
		B: uint8(float64(c1.B) + (float64(c2.B)-float64(c1.B))*t),
	}
}

// GradientText applies a gradient to text
func GradientText(text string, colors []RGB) string {
	if len(text) == 0 || len(colors) == 0 {
		return text
	}

	result := ""
	textLen := len([]rune(text))

	for i, char := range text {
		// Calculate position in gradient (0.0 to 1.0)
		t := float64(i) / float64(textLen-1)

		// Find which two colors to interpolate between
		segmentSize := 1.0 / float64(len(colors)-1)
		segment := int(t / segmentSize)
		if segment >= len(colors)-1 {
			segment = len(colors) - 2
		}

		// Local interpolation within the segment
		localT := (t - float64(segment)*segmentSize) / segmentSize

		color := Lerp(colors[segment], colors[segment+1], localT)
		result += color.ToANSI() + string(char)
	}

	return result + "\x1b[0m"
}

// Reset ANSI color
const Reset = "\x1b[0m"

// ClearScreen clears the terminal
func ClearScreen() string {
	return "\x1b[2J\x1b[H"
}

// HideCursor hides the terminal cursor
func HideCursor() string {
	return "\x1b[?25l"
}

// ShowCursor shows the terminal cursor
func ShowCursor() string {
	return "\x1b[?25h"
}

// MoveCursor moves the cursor to x, y (1-indexed)
func MoveCursor(x, y int) string {
	return fmt.Sprintf("\x1b[%d;%dH", y, x)
}

// Styles for lipgloss
var (
	TitleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(ElectricRGB)

	SubtitleStyle = lipgloss.NewStyle().
			Foreground(TealRGB).
			Italic(true)

	ActiveStyle = lipgloss.NewStyle().
			Foreground(CyanRGB).
			Bold(true)

	InactiveStyle = lipgloss.NewStyle().
			Foreground(DimGray)

	SelectedStyle = lipgloss.NewStyle().
			Foreground(ElectricRGB).
			Bold(true).
			Background(lipgloss.Color("#1A1F2E"))

	StatusStyle = lipgloss.NewStyle().
			Foreground(GreenRGB).
			Italic(true)

	BorderStyle = lipgloss.NewStyle().
			Foreground(TealRGB).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(TealRGB).
			Padding(1, 2)
)
