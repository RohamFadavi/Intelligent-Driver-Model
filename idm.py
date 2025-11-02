import math

def idm_accel(v, gap, rel_vel, v0=1.2, a=0.6, b=0.8, T=1.0, s0=0.2, delta=4):
    """
    IDM longitudinal acceleration.
    v       = follower speed [m/s]
    gap     = distance to leader/stop line [m]  (must be > 0)
    rel_vel = v_follower - v_leader [m/s]  (positive if we are faster/closing in)

    Returns: acceleration [m/s^2]
    """
    gap = max(gap, 1e-3)
    s_star = s0 + v*T + (v * rel_vel) / (2.0 * math.sqrt(a * b))
    return a * (1.0 - (v / max(v0, 1e-6))**delta - (s_star / gap)**2)


class IDM:
    def __init__(self, dt=0.1, v0=1.2, a=0.6, b=0.8, T=1.0, s0=0.2, delta=4):
        self.dt, self.v0, self.a, self.b, self.T, self.s0, self.delta = dt, v0, a, b, T, s0, delta

    def accel(self, v, gap, rel_vel):
        return idm_accel(v, gap, rel_vel, self.v0, self.a, self.b, self.T, self.s0, self.delta)

    def step(self, v, gap, rel_vel, clip_to_v0=True):
        a_idm = self.accel(v, gap, rel_vel)
        v_next = v + a_idm * self.dt
        if clip_to_v0:
            v_next = max(0.0, min(self.v0, v_next))
        else:
            v_next = max(0.0, v_next)
        return v_next, a_idm