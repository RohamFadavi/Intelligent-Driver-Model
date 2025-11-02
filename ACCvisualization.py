
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

@dataclass
class IDMParams:
    v0: float = 1.0
    T: float = 1.2
    s0: float = 0.5
    a_max: float = 1.0
    b_comf: float = 1.5
    delta: float = 4.0

@dataclass
class SimParams:
    dt: float = 0.05
    speed_limit: float = 1.2
    leader_v: float = 0.9
    init_gap: float = 5.0

class IDMVis:
    def __init__(self, idm: IDMParams, sim: SimParams):
        self.idm = idm
        self.sim = sim
        self.reset_state()
        self.fig, self.ax = plt.subplots(figsize=(9, 4))
        plt.subplots_adjust(left=0.07, right=0.98, bottom=0.35)
        self.ax.set_title("IDM Leader-Follower (1D)")
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylim(-0.5, 0.5)
        self.ax.set_xlim(-1, 30)
        self.ax.get_yaxis().set_visible(False)
        self.ax.grid(True, alpha=0.3)
        (self.lead_pt,) = self.ax.plot([], [], 'o', color='green', label='Leader')
        (self.fol_pt,) = self.ax.plot([], [], 'o', color='blue', label='Follower')
        (self.fol_trail,) = self.ax.plot([], [], '-', color='blue', alpha=0.4, lw=2, label='Follower path')
        self.ax.legend(loc='upper left')
        self.info = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, va='top')
        self._build_sliders()
        self.running = True
        ax_play = plt.axes([0.07, 0.08, 0.12, 0.05])
        ax_reset = plt.axes([0.22, 0.08, 0.12, 0.05])
        self.btn_play = Button(ax_play, 'Pause')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_play.on_clicked(self._toggle)
        self.btn_reset.on_clicked(self._reset)
        self.anim = FuncAnimation(self.fig, self._animate, interval=int(self.sim.dt * 1000), blit=False)

    def reset_state(self):
        self.t = 0.0
        self.lead_x = self.sim.init_gap
        self.lead_v = self.sim.leader_v
        self.fol_x = 0.0
        self.fol_v = 0.0
        self.fx_hist = []
        self.fy_hist = []

    @staticmethod
    def idm_acc(v: float, s: float, dv: float, p: IDMParams) -> float:
        s = max(1e-3, s)
        v0 = max(1e-6, p.v0)
        a = max(1e-6, p.a_max)
        b = max(1e-6, p.b_comf)
        s_star = p.s0 + v * p.T + (v * dv) / (2.0 * math.sqrt(a * b))
        return a * (1.0 - (v / v0) ** p.delta - (s_star / s) ** 2)

    def step(self):
        self.lead_v = self.sim.leader_v
        self.lead_x += self.lead_v * self.sim.dt
        s = self.lead_x - self.fol_x
        dv = self.fol_v - self.lead_v
        acc = self.idm_acc(self.fol_v, s, dv, self.idm)
        self.fol_v = max(0.0, min(self.sim.speed_limit, self.fol_v + acc * self.sim.dt))
        self.fol_x += self.fol_v * self.sim.dt
        self.t += self.sim.dt

    def _build_sliders(self):
        ax_v0   = plt.axes([0.07, 0.28, 0.35, 0.03])
        ax_T    = plt.axes([0.07, 0.24, 0.35, 0.03])
        ax_s0   = plt.axes([0.07, 0.20, 0.35, 0.03])
        ax_a    = plt.axes([0.07, 0.16, 0.35, 0.03])
        ax_b    = plt.axes([0.07, 0.12, 0.35, 0.03])
        ax_lv   = plt.axes([0.55, 0.28, 0.35, 0.03])
        ax_vmax = plt.axes([0.55, 0.24, 0.35, 0.03])
        ax_gap  = plt.axes([0.55, 0.20, 0.35, 0.03])
        self.sl_v0 = Slider(ax_v0,   'v0 [m/s]', 0.2, 1.5, valinit=self.idm.v0)
        self.sl_T  = Slider(ax_T,    'T [s]',    0.5, 2.5, valinit=self.idm.T)
        self.sl_s0 = Slider(ax_s0,   's0 [m]',   0.1, 2.0, valinit=self.idm.s0)
        self.sl_a  = Slider(ax_a,    'a_max',    0.2, 2.0, valinit=self.idm.a_max)
        self.sl_b  = Slider(ax_b,    'b_comf',   0.2, 3.0, valinit=self.idm.b_comf)
        self.sl_lv = Slider(ax_lv,   'leader v', 0.0, 1.5, valinit=self.sim.leader_v)
        self.sl_vm = Slider(ax_vmax, 'speed lim',0.4, 1.6, valinit=self.sim.speed_limit)
        self.sl_gp = Slider(ax_gap,  'init gap', 1.0, 15.0, valinit=self.sim.init_gap)
        def on_change(_):
            self.idm.v0 = float(self.sl_v0.val)
            self.idm.T = float(self.sl_T.val)
            self.idm.s0 = float(self.sl_s0.val)
            self.idm.a_max = float(self.sl_a.val)
            self.idm.b_comf = float(self.sl_b.val)
            self.sim.leader_v = float(self.sl_lv.val)
            self.sim.speed_limit = float(self.sl_vm.val)
            self.sim.init_gap = float(self.sl_gp.val)
        for s in (self.sl_v0, self.sl_T, self.sl_s0, self.sl_a, self.sl_b, self.sl_lv, self.sl_vm, self.sl_gp):
            s.on_changed(on_change)

    def _toggle(self, _):
        self.running = not self.running
        self.btn_play.label.set_text('Pause' if self.running else 'Play')

    def _reset(self, _):
        self.reset_state()

    def _animate(self, _):
        if self.running:
            self.step()
        self.fx_hist.append(self.fol_x)
        self.fy_hist.append(0.0)
        if len(self.fx_hist) > 400:
            self.fx_hist = self.fx_hist[-400:]
            self.fy_hist = self.fy_hist[-400:]
        x_min = min(self.fol_x, self.lead_x) - 2.0
        x_max = max(self.fol_x, self.lead_x) + 5.0
        cur_min, cur_max = self.ax.get_xlim()
        if x_min <= cur_min + 1 or x_max >= cur_max - 1:
            self.ax.set_xlim(x_min, x_max)
        self.lead_pt.set_data([self.lead_x], [0.0])
        self.fol_pt.set_data([self.fol_x], [0.0])
        self.fol_trail.set_data(self.fx_hist, self.fy_hist)
        gap = max(0.0, self.lead_x - self.fol_x)
        self.info.set_text(
            f"t={self.t:4.1f}s  gap={gap:4.2f}m  v_lead={self.lead_v:3.2f} m/s  v_fol={self.fol_v:3.2f} m/s\n"
            f"v0={self.idm.v0:.2f}  T={self.idm.T:.2f}  s0={self.idm.s0:.2f}  a={self.idm.a_max:.2f}  b={self.idm.b_comf:.2f}  lim={self.sim.speed_limit:.2f}"
        )
        return self.lead_pt, self.fol_pt, self.fol_trail, self.info

def main():
    idm = IDMParams()
    sim = SimParams()
    app = IDMVis(idm, sim)
    plt.show()

if __name__ == '__main__':
    main()
