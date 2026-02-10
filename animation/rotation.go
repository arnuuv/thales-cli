package animation

import (
	"time"
)

// RotationState manages the animation state for rotating objects
type RotationState struct {
	MapFrame       int
	ShipFrame      int
	StartTime      time.Time
	CurrentTime    time.Time
	ActiveObject   string // "map" or "ship"
	LastSwitchTime time.Time
}

// NewRotationState creates a new rotation state
func NewRotationState() *RotationState {
	now := time.Now()
	return &RotationState{
		MapFrame:       0,
		ShipFrame:      0,
		StartTime:      now,
		CurrentTime:    now,
		ActiveObject:   "map",
		LastSwitchTime: now,
	}
}

// Update updates the rotation state based on elapsed time
func (r *RotationState) Update(currentTime time.Time) {
	r.CurrentTime = currentTime

	// No frame animation needed - objects are static
	r.MapFrame = 0
	r.ShipFrame = 0

	// Switch active object every 5 seconds for brightness toggle
	if currentTime.Sub(r.LastSwitchTime).Seconds() >= 5.0 {
		if r.ActiveObject == "map" {
			r.ActiveObject = "ship"
		} else {
			r.ActiveObject = "map"
		}
		r.LastSwitchTime = currentTime
	}
}

// IsMapActive returns true if the map is the active object
func (r *RotationState) IsMapActive() bool {
	return r.ActiveObject == "map"
}

// IsShipActive returns true if the ship is the active object
func (r *RotationState) IsShipActive() bool {
	return r.ActiveObject == "ship"
}
