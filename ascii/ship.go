package ascii

import "strings"

var ShipFrames = []string{
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

func GetShipFrame(frameIndex int, colorFunc func(string) string) string {
	if len(ShipFrames) == 0 {
		return ""
	}
	frame := ShipFrames[frameIndex%len(ShipFrames)]
	
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

func ShipFrameCount() int {
	return len(ShipFrames)
}
