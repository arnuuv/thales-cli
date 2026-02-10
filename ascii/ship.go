package ascii

import "strings"

// ShipFrames contains ASCII art frames for ship display
var ShipFrames = []string{
	// Modern Maritime Vessel
	"               /|___\n" +
		"             ///|   ))\n" +
		"           /////|   )))\n" +
		"         ///////|    )))\n" +
		"       /////////|     )))\n" +
		"     ///////////|     ))))\n" +
		"   /////////////|     )))\n" +
		"  //////////////|    )))\n" +
		"////////////////|___))\n" +
		"  ______________|________\n" +
		"  \\                    /\n" +
		"~~~~~~~~~~~~~~~~~~~~~~~~~~~",
}

// GetShipFrame returns the frame for the given index with color applied
func GetShipFrame(frameIndex int, colorFunc func(string) string) string {
	if len(ShipFrames) == 0 {
		return ""
	}
	frame := ShipFrames[frameIndex%len(ShipFrames)]
	
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

// ShipFrameCount returns the number of frames for ship animation
func ShipFrameCount() int {
	return len(ShipFrames)
}
