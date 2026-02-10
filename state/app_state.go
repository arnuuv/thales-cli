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
	SelectedDomain Domain
	SelectedOption string
	Status         string
	CursorPos      int
	Options        []Option
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
