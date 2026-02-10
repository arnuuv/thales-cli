package ascii

import "strings"

// MapFrames contains ASCII art frames for map display
var MapFrames = []string{
	// Globe - Large
	"                 _,--',   _._.--._____\n" +
		"         .--.--';_'-.', \";_      _.,-'\n" +
		"       .'--'.  _.'    {`'-;_ .-.>.'\n" +
		"             '-:_      )  / `' '=.\n" +
		"               ) >     {_/,     /~)\n" +
		"               |/               `^ .'\n" +
		"\n" +
		"         ╔═══════════════════════╗\n" +
		"         ║   GEOSPATIAL  CORE    ║\n" +
		"         ║                       ║\n" +
		"         ║       ░▒▒▓▓▓▓▓▓▒▒░    ║\n" +
		"         ║      ░▒▓████████▓▒░   ║\n" +
		"         ║      ░▒▓████████▓▒░   ║\n" +
		"         ║       ░░▒▒▓▓▓▓▒▒░░    ║\n" +
		"         ║                       ║\n" +
		"         ╚═══════════════════════╝",
}

// GetMapFrame returns the frame for the given index with color applied
func GetMapFrame(frameIndex int, colorFunc func(string) string) string {
	if len(MapFrames) == 0 {
		return ""
	}
	frame := MapFrames[frameIndex%len(MapFrames)]
	
	// Apply color to each line individually to ensure it persists
	lines := strings.Split(frame, "\n")
	var coloredLines []string
	for _, line := range lines {
		if line != "" {
			coloredLines = append(coloredLines, colorFunc(line))
		} else {
			coloredLines = append(coloredLines, "")
		}
	}
	return strings.Join(coloredLines, "\n")
}

// MapFrameCount returns the number of frames for map animation
func MapFrameCount() int {
	return len(MapFrames)
}
