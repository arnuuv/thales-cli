package animation

import (
	"time"
)

type RotationState struct {
	MapFrame       int
	ShipFrame      int
	StartTime      time.Time
	CurrentTime    time.Time
	ActiveObject   string
	LastSwitchTime time.Time
}

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

func (r *RotationState) Update(currentTime time.Time) {
	r.CurrentTime = currentTime
	r.MapFrame = 0
	r.ShipFrame = 0

	if currentTime.Sub(r.LastSwitchTime).Seconds() >= 5.0 {
		if r.ActiveObject == "map" {
			r.ActiveObject = "ship"
		} else {
			r.ActiveObject = "map"
		}
		r.LastSwitchTime = currentTime
	}
}

func (r *RotationState) IsMapActive() bool {
	return r.ActiveObject == "map"
}

func (r *RotationState) IsShipActive() bool {
	return r.ActiveObject == "ship"
}
