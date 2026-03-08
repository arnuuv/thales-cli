package ui

import (
	"strings"

	"github.com/arnavanchan/thales-cli/animation"
	"github.com/arnavanchan/thales-cli/ascii"
	"github.com/arnavanchan/thales-cli/state"
	"github.com/charmbracelet/lipgloss"
)

const (
	minWidth  = 80
	minHeight = 30
)

// Layout manages the main UI layout
type Layout struct {
	Width  int
	Height int
}

// NewLayout creates a new layout
func NewLayout(width, height int) *Layout {
	if width < minWidth {
		width = minWidth
	}
	if height < minHeight {
		height = minHeight
	}

	return &Layout{
		Width:  width,
		Height: height,
	}
}

// Render renders the complete UI
func (l *Layout) Render(appState *state.AppState, rotation *animation.RotationState) string {
	var sections []string

	// Title with gradient
	title := "THALES FICTITIOUS GENERATOR"
	gradientTitle := GradientText(title, []RGB{Cyan, Teal, Purple, Electric})

	subtitle := "Synthetic Geospatial & Maritime Imagery"
	gradientSubtitle := Teal.Dim(0.8).ToANSI() + subtitle + Reset

	sections = append(sections, gradientTitle)
	sections = append(sections, gradientSubtitle)
	sections = append(sections, l.separator())
	sections = append(sections, "")

	// Display map only (centered)
	mapFrame := l.getColoredMapFrame(rotation)
	sections = append(sections, mapFrame)
	sections = append(sections, l.separator())
	sections = append(sections, "")

	// Status section
	sections = append(sections, l.renderStatus(appState))
	sections = append(sections, "")

	// Menu options
	sections = append(sections, l.renderMenu(appState))

	// Join all sections
	content := strings.Join(sections, "\n")

	// Use lipgloss to center everything
	style := lipgloss.NewStyle().
		Width(l.Width).
		Height(l.Height).
		Align(lipgloss.Center, lipgloss.Center)

	return style.Render(content)
}

// separator creates a horizontal line
func (l *Layout) separator() string {
	return strings.Repeat("─", 100)
}

// renderSideBySideObjects renders the map and ship objects side by side
func (l *Layout) renderSideBySideObjects(mapFrame, shipFrame string) string {
	mapLines := strings.Split(mapFrame, "\n")
	shipLines := strings.Split(shipFrame, "\n")

	maxLines := len(mapLines)
	if len(shipLines) > maxLines {
		maxLines = len(shipLines)
	}

	// Pad arrays to same length
	for len(mapLines) < maxLines {
		mapLines = append(mapLines, "")
	}
	for len(shipLines) < maxLines {
		shipLines = append(shipLines, "")
	}

	var lines []string
	spacing := "          " // Space between objects

	for i := 0; i < maxLines; i++ {
		line := mapLines[i] + spacing + shipLines[i]
		lines = append(lines, line)
	}

	return strings.Join(lines, "\n")
}

// renderStatus renders the status section
func (l *Layout) renderStatus(appState *state.AppState) string {
	var status strings.Builder

	mode := Green.ToANSI() + "Mode:" + Reset + " UI Prototype"
	statusText := Cyan.ToANSI() + "Status:" + Reset + " " + appState.Status
	if appState.GeneratingMap || appState.GeneratingMaritime {
		statusText = Cyan.ToANSI() + "Status:" + Reset + " " + Green.ToANSI() + "Generating... (see terminal)" + Reset
	}

	domain := Purple.ToANSI() + "Selected Domain:" + Reset + " "
	switch appState.SelectedDomain {
	case state.DomainTerrain:
		domain += Teal.ToANSI() + "Terrain & Map Imagery" + Reset
	case state.DomainMaritime:
		domain += Electric.ToANSI() + "Maritime Imagery" + Reset
	default:
		domain += Gray.ToANSI() + "None" + Reset
	}

	status.WriteString("  " + mode + "\n")
	status.WriteString("  " + statusText + "\n")
	status.WriteString("  " + domain + "\n")

	return status.String()
}

// renderMenu renders the menu options
func (l *Layout) renderMenu(appState *state.AppState) string {
	var menu strings.Builder

	// Show terrain selection state
	if appState.TerrainSelectionStep == "resolution" {
		menu.WriteString("  " + Cyan.ToANSI() + "Select Resolution(s):" + Reset + "\n")
		menu.WriteString("      " + getCheckbox(appState.HasResolution("1m")) + " [1] 1m Resolution\n")
		menu.WriteString("      " + getCheckbox(appState.HasResolution("10m")) + " [2] 10m Resolution\n")
		menu.WriteString("      " + getCheckbox(appState.HasResolution("25cm")) + " [3] 25cm Resolution\n")
		menu.WriteString("      " + Teal.Dim(0.7).ToANSI() + "[4] All Resolutions" + Reset + "\n")
		menu.WriteString("      " + Green.ToANSI() + "[Enter] Continue" + Reset + "\n")
		menu.WriteString("\n")
	} else if appState.TerrainSelectionStep == "area" {
		menu.WriteString("  " + Cyan.ToANSI() + "Select Area Size:" + Reset + "\n")
		menu.WriteString("      " + getCheckbox(appState.SelectedAreaKm == 1.0) + " [1] 1km x 1km\n")
		menu.WriteString("      " + getCheckbox(appState.SelectedAreaKm == 2.0) + " [2] 2km x 2km\n")
		menu.WriteString("      " + Green.ToANSI() + "[Enter] Generate" + Reset + "\n")
		menu.WriteString("\n")
		menu.WriteString("  " + Teal.Dim(0.7).ToANSI() + "Selected resolutions: " + formatResolutions(appState.SelectedResolutions) + Reset + "\n")
		menu.WriteString("\n")
	} else {
		// Normal menu
		menu.WriteString("  " + Cyan.ToANSI() + "[1]" + Reset + " Terrain & Map Imagery\n")
		menu.WriteString("      " + Teal.Dim(0.7).ToANSI() + "- 1m Resolution" + Reset + "\n")
		menu.WriteString("      " + Teal.Dim(0.7).ToANSI() + "- 10m Resolution" + Reset + "\n")
		menu.WriteString("      " + Teal.Dim(0.7).ToANSI() + "- 25cm Resolution" + Reset + "\n")
		menu.WriteString("\n")

		menu.WriteString("  " + Electric.ToANSI() + "[2]" + Reset + " Maritime Imagery\n")
		menu.WriteString("      " + Electric.Dim(0.7).ToANSI() + "- Vessels" + Reset + "\n")
		menu.WriteString("      " + Electric.Dim(0.7).ToANSI() + "- Sea Lanes" + Reset + "\n")
		menu.WriteString("      " + Electric.Dim(0.7).ToANSI() + "- Coastal Assets" + Reset + "\n")
		menu.WriteString("      " + Electric.Dim(0.7).ToANSI() + "- Produce GeoJSON routes" + Reset + "\n")
		menu.WriteString("\n")
	}

	menu.WriteString("  " + Teal.Dim(0.8).ToANSI() + "[D] Download  [Q] Quit" + Reset + "\n")
	
	if appState.LastDatasetZipPath != "" {
		menu.WriteString("  " + Green.Dim(0.8).ToANSI() + "Latest Dataset: " + appState.LastDatasetZipPath + Reset + "\n")
	} else if appState.LastMapOutputDir != "" || appState.LastMaritimeGeoJSONPath != "" {
		menu.WriteString("  " + Green.Dim(0.8).ToANSI() + "Output: ")
		if appState.LastMapOutputDir != "" {
			menu.WriteString(appState.LastMapOutputDir)
		}
		if appState.LastMaritimeGeoJSONPath != "" {
			if appState.LastMapOutputDir != "" {
				menu.WriteString("  |  ")
			}
			menu.WriteString(appState.LastMaritimeGeoJSONPath)
		}
		menu.WriteString(Reset + "\n")
	}

	return menu.String()
}

// getCheckbox returns a checkbox character based on selection state
func getCheckbox(selected bool) string {
	if selected {
		return Green.ToANSI() + "[✓]" + Reset
	}
	return Gray.ToANSI() + "[ ]" + Reset
}

// formatResolutions formats the resolution list for display
func formatResolutions(resolutions []string) string {
	if len(resolutions) == 0 {
		return "none"
	}
	result := ""
	for i, res := range resolutions {
		if i > 0 {
			result += ", "
		}
		result += res
	}
	return result
}

// centerText centers text accounting for ANSI codes
func centerText(text string, width int, visualLen int) string {
	if visualLen >= width {
		return text
	}
	leftPadding := (width - visualLen) / 2
	return strings.Repeat(" ", leftPadding) + text
}

// centerTextRaw centers raw text (for lines that may already have ANSI)
func centerTextRaw(text string, width int) string {
	// Count visual length (rough approximation)
	visualLen := countVisualLength(text)
	if visualLen >= width {
		return text
	}
	leftPadding := (width - visualLen) / 2
	if leftPadding < 0 {
		leftPadding = 0
	}
	return strings.Repeat(" ", leftPadding) + text
}

// countVisualLength counts the visual length of a string (removing ANSI codes)
func countVisualLength(s string) int {
	inEscape := false
	count := 0

	for _, ch := range s {
		if ch == '\x1b' {
			inEscape = true
			continue
		}
		if inEscape {
			if (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') {
				inEscape = false
			}
			continue
		}
		count++
	}

	return count
}

// RenderStartupTitle renders the startup title screen
func RenderStartupTitle(width, height int) string {
	title := `████████╗██╗  ██╗ █████╗ ██╗     ███████╗███████╗
╚══██╔══╝██║  ██║██╔══██╗██║     ██╔════╝██╔════╝
   ██║   ███████║███████║██║     █████╗  ███████╗
   ██║   ██╔══██║██╔══██║██║     ██╔══╝  ╚════██║
   ██║   ██║  ██║██║  ██║███████╗███████╗███████║
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝

███████╗██╗ ██████╗████████╗██╗████████╗██╗ ██████╗ ██╗   ██╗███████╗
██╔════╝██║██╔════╝╚══██╔══╝██║╚══██╔══╝██║██╔═══██╗██║   ██║██╔════╝
█████╗  ██║██║        ██║   ██║   ██║   ██║██║   ██║██║   ██║███████╗
██╔══╝  ██║██║        ██║   ██║   ██║   ██║██║   ██║██║   ██║╚════██║
██║     ██║╚██████╗   ██║   ██║   ██║   ██║╚██████╔╝╚██████╔╝███████║
╚═╝     ╚═╝ ╚═════╝   ╚═╝   ╚═╝   ╚═╝   ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝

 ██████╗ ███████╗███╗   ██╗███████╗██████╗  █████╗ ████████╗ ██████╗ ██████╗ 
██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗
██║  ███╗█████╗  ██╔██╗ ██║█████╗  ██████╔╝███████║   ██║   ██║   ██║██████╔╝
██║   ██║██╔══╝  ██║╚██╗██║██╔══╝  ██╔══██╗██╔══██║   ██║   ██║   ██║██╔══██╗
╚██████╔╝███████╗██║ ╚████║███████╗██║  ██║██║  ██║   ██║   ╚██████╔╝██║  ██║
 ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝`

	// Apply gradient to title
	lines := strings.Split(title, "\n")
	var coloredLines []string

	colors := []RGB{Cyan, Teal, Purple, Electric}

	for i, line := range lines {
		// Calculate color for this line
		t := float64(i) / float64(len(lines)-1)
		colorIndex := int(t * float64(len(colors)-1))
		if colorIndex >= len(colors)-1 {
			colorIndex = len(colors) - 2
		}
		localT := (t - float64(colorIndex)/float64(len(colors)-1)) * float64(len(colors)-1)

		lineColor := Lerp(colors[colorIndex], colors[colorIndex+1], localT)
		coloredLines = append(coloredLines, lineColor.ToANSI()+line+Reset)
	}

	titleBlock := strings.Join(coloredLines, "\n")

	subtitle := Teal.Dim(0.8).ToANSI() + "Synthetic Geospatial & Maritime Imagery System" + Reset
	initializing := Green.Dim(0.7).ToANSI() + "Initializing..." + Reset

	content := titleBlock + "\n\n" + subtitle + "\n\n" + initializing

	// Use lipgloss to center everything
	style := lipgloss.NewStyle().
		Width(width).
		Height(height).
		Align(lipgloss.Center, lipgloss.Center)

	return style.Render(content)
}

// GetTerminalSize returns a reasonable default if not detectable
func GetTerminalSize() (int, int) {
	return 120, 40
}

// getColoredMapFrame returns the map frame with appropriate coloring
func (l *Layout) getColoredMapFrame(rotation *animation.RotationState) string {
	colorFunc := func(s string) string {
		// Keep at constant bright teal - no dimming
		return Teal.ToANSI() + s + Reset
	}

	// Always use frame 0 (static)
	return ascii.GetMapFrame(0, colorFunc)
}

// getColoredShipFrame returns the ship frame with appropriate coloring
func (l *Layout) getColoredShipFrame(rotation *animation.RotationState) string {
	colorFunc := func(s string) string {
		if rotation.IsShipActive() {
			return Electric.Brighten(1.2).ToANSI() + s + Reset
		}
		return Electric.Dim(0.4).ToANSI() + s + Reset
	}

	// Always use frame 0 (static)
	return ascii.GetShipFrame(0, colorFunc)
}
