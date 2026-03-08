package state

// Domain represents the two main domains
type Domain string

const (
	DomainNone     Domain = "none"
	DomainTerrain  Domain = "terrain"
	DomainMaritime Domain = "maritime"
)

// Option represents a selectable option
type Option struct {
	Key         string
	Label       string
	SubOptions  []string
	Domain      Domain
	IsSubOption bool
	ParentKey   string
}

// AppState holds the application state
type AppState struct {
	SelectedDomain         Domain
	SelectedOption         string
	Status                 string
	CursorPos              int
	Options                []Option
	LastMapOutputDir       string   // After map generation
	LastMaritimeGeoJSONPath string  // After maritime GeoJSON production
	GeneratingMap          bool     // True while map generation runs
	GeneratingMaritime     bool     // True while maritime GeoJSON runs
}

// NewAppState creates a new application state
func NewAppState() *AppState {
	return &AppState{
		SelectedDomain: DomainNone,
		SelectedOption: "",
		Status:         "Idle",
		CursorPos:      0,
		Options: []Option{
			{
				Key:    "1",
				Label:  "Terrain & Map Imagery",
				Domain: DomainTerrain,
				SubOptions: []string{
					"1m Resolution",
					"10m Resolution",
					"25cm Resolution",
				},
			},
			{
				Key:    "1a",
				Label:  "  - 1m Resolution",
				Domain: DomainTerrain,
				IsSubOption: true,
				ParentKey: "1",
			},
			{
				Key:    "1b",
				Label:  "  - 10m Resolution",
				Domain: DomainTerrain,
				IsSubOption: true,
				ParentKey: "1",
			},
			{
				Key:    "1c",
				Label:  "  - 25cm Resolution",
				Domain: DomainTerrain,
				IsSubOption: true,
				ParentKey: "1",
			},
			{
				Key:    "2",
				Label:  "Maritime Imagery",
				Domain: DomainMaritime,
				SubOptions: []string{
					"Vessels",
					"Sea Lanes",
					"Coastal Assets",
				},
			},
			{
				Key:    "2a",
				Label:  "  - Vessels",
				Domain: DomainMaritime,
				IsSubOption: true,
				ParentKey: "2",
			},
			{
				Key:    "2b",
				Label:  "  - Sea Lanes",
				Domain: DomainMaritime,
				IsSubOption: true,
				ParentKey: "2",
			},
			{
				Key:    "2c",
				Label:  "  - Coastal Assets",
				Domain: DomainMaritime,
				IsSubOption: true,
				ParentKey: "2",
			},
			{
				Key:         "2d",
				Label:       "  - Produce GeoJSON routes",
				Domain:      DomainMaritime,
				IsSubOption: true,
				ParentKey:   "2",
			},
		},
	}
}

// SelectOption updates the state based on selected option
func (s *AppState) SelectOption(key string) {
	for _, opt := range s.Options {
		if opt.Key == key {
			s.SelectedDomain = opt.Domain
			s.SelectedOption = opt.Label

			// Update status message
			if opt.IsSubOption {
				parentLabel := ""
				for _, parent := range s.Options {
					if parent.Key == opt.ParentKey {
						parentLabel = parent.Label
						break
					}
				}
				s.Status = "Selected " + parentLabel + " → " + opt.Label
			} else {
				s.Status = "Selected " + opt.Label
			}
			return
		}
	}
}

// MoveSelection moves the cursor up or down
func (s *AppState) MoveSelection(delta int) {
	s.CursorPos += delta
	if s.CursorPos < 0 {
		s.CursorPos = len(s.Options) - 1
	} else if s.CursorPos >= len(s.Options) {
		s.CursorPos = 0
	}
}

// GetCurrentOption returns the currently selected option
func (s *AppState) GetCurrentOption() Option {
	if s.CursorPos >= 0 && s.CursorPos < len(s.Options) {
		return s.Options[s.CursorPos]
	}
	return Option{}
}

// ResolutionForMapOption returns the resolution string for map option keys (1a->1m, 1b->10m, 1c->25cm).
// Returns "" if not a map resolution option.
func ResolutionForMapOption(key string) string {
	switch key {
	case "1a":
		return "1m"
	case "1b":
		return "10m"
	case "1c":
		return "25cm"
	default:
		return ""
	}
}

// IsMapResolutionOption returns true for 1a, 1b, 1c
func IsMapResolutionOption(key string) bool {
	return ResolutionForMapOption(key) != ""
}

// IsMaritimeGeoJSONOption returns true for 2d
func IsMaritimeGeoJSONOption(key string) bool {
	return key == "2d"
}
