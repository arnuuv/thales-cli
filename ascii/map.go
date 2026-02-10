package ascii

import "strings"

var MapFrames = []string{
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

func GetMapFrame(frameIndex int, colorFunc func(string) string) string {
	if len(MapFrames) == 0 {
		return ""
	}
	frame := MapFrames[frameIndex%len(MapFrames)]
	
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

func MapFrameCount() int {
	return len(MapFrames)
}
