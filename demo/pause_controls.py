import pybullet as p

class PauseControls:
    def __init__(self):
        self.paused = False
        self.space_key = ord(' ')
        self.last_space_pressed = False
        
    def update(self):
        """Check if spacebar is pressed to toggle pause"""
        keys = p.getKeyboardEvents()
        
        if self.space_key in keys:
            key_state = keys[self.space_key]
            
            # Toggle on key press (not hold)
            if key_state == p.KEY_WAS_TRIGGERED and not self.last_space_pressed:
                self.paused = not self.paused
                self.last_space_pressed = True
                if self.paused:
                    print("\n⏸  PAUSED - Press SPACEBAR to resume")
                else:
                    print("▶  RESUMED\n")
            elif key_state == p.KEY_WAS_RELEASED:
                self.last_space_pressed = False
        
        return self.paused