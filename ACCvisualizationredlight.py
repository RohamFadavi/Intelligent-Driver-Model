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
    init_gap: float = 8.0
    light_x: float = 15.0
    start_red: bool = True
class IDMVisRedLight:
    def __init__(self, idm: IDMParams, sim: SimParams):
        self.idm = idm
        self.sim = sim
        self.reset_state()
        self.fig, self.ax = plt.subplots(figsize=(9, 4))
        plt.subplots_adjust(left=0.07, right=0.98, bottom=0.35)
        self.ax.set_title("IDM Follower vs Red Light (1D)")
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylim(-0.5, 0.5)
        self.ax.set_xlim(-1, 30)
        self.ax.get_yaxis().set_visible(False)
        self.ax.grid(True, alpha=0.3)
        (self.fol_pt,) = self.ax.plot([], [], 'o', color='blue', label='Follower')
        (self.fol_trail,) = self.ax.plot([], [], '-', color='blue', alpha=0.4, lw=2, label='Follower path')
        (self.light_pt,) = self.ax.plot([], [], 's', ms=10, label='Light')
        self.light_line = self.ax.axvline(self.sim.light_x, color='k', ls='--', lw=1, alpha=0.5)
        self.ax.legend(loc='upper left')
        self.info = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, va='top')
        self._build_sliders()
        self.running = True
        ax_play = plt.axes([0.07, 0.08, 0.12, 0.05])
        ax_reset = plt.axes([0.22, 0.08, 0.12, 0.05])
        ax_toggle = plt.axes([0.37, 0.08, 0.16, 0.05])
        self.btn_play = Button(ax_play, 'Pause')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_toggle = Button(ax_toggle, 'Toggle RED/GREEN')
        self.btn_play.on_clicked(self._toggle)
        self.btn_reset.on_clicked(self._reset)
        self.btn_toggle.on_clicked(self._toggle_light)
        self.anim = FuncAnimation(self.fig, self._animate, interval=int(self.sim.dt * 1000), blit=False)
    def reset_state(self):
        self.t = 0.0
        self.fol_x = 0.0
        self.fol_v = 0.0
        self.light_x = float(self.sim.light_x)
        self.light_red = bool(self.sim.start_red)
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
        if self.light_red:
            s = self.light_x - self.fol_x
            dv = self.fol_v - 0.0
        else:
            s = 1e9
            dv = 0.0
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
        ax_vmax = plt.axes([0.55, 0.28, 0.35, 0.03])
        ax_lpos = plt.axes([0.55, 0.24, 0.35, 0.03])
        ax_gap  = plt.axes([0.55, 0.20, 0.35, 0.03])
        self.sl_v0 = Slider(ax_v0,   'v0 [m/s]', 0.2, 1.5, valinit=self.idm.v0)
        self.sl_T  = Slider(ax_T,    'T [s]',    0.5, 2.5, valinit=self.idm.T)
        self.sl_s0 = Slider(ax_s0,   's0 [m]',   0.1, 2.0, valinit=self.idm.s0)
        self.sl_a  = Slider(ax_a,    'a_max',    0.2, 2.0, valinit=self.idm.a_max)
        self.sl_b  = Slider(ax_b,    'b_comf',   0.2, 3.0, valinit=self.idm.b_comf)
        self.sl_vm = Slider(ax_vmax, 'speed lim',0.4, 1.6, valinit=self.sim.speed_limit)
        self.sl_lp = Slider(ax_lpos, 'light x',  5.0, 25.0, valinit=self.sim.light_x)
        self.sl_gp = Slider(ax_gap,  'init gap', 1.0, 15.0, valinit=self.sim.init_gap)
        def on_change(_):
            self.idm.v0 = float(self.sl_v0.val)
            self.idm.T = float(self.sl_T.val)
            self.idm.s0 = float(self.sl_s0.val)
            self.idm.a_max = float(self.sl_a.val)
            self.idm.b_comf = float(self.sl_b.val)
            self.sim.speed_limit = float(self.sl_vm.val)
            self.light_x = float(self.sl_lp.val)
            self.light_line.set_xdata([self.light_x, self.light_x])
            self.sim.init_gap = float(self.sl_gp.val)
        for s in (self.sl_v0, self.sl_T, self.sl_s0, self.sl_a, self.sl_b, self.sl_vm, self.sl_lp, self.sl_gp):
            s.on_changed(on_change)
    def _toggle(self, _):
        self.running = not self.running
        self.btn_play.label.set_text('Pause' if self.running else 'Play')
    def _reset(self, _):
        self.sim.light_x = float(self.sl_lp.val)
        self.reset_state()
    def _toggle_light(self, _):
        self.light_red = not self.light_red
    def _animate(self, _):
        if self.running:
            if self.t == 0.0:
                self.fol_x = max(0.0, self.light_x - self.sim.init_gap)
            self.step()
        self.fx_hist.append(self.fol_x)
        self.fy_hist.append(0.0)
        if len(self.fx_hist) > 400:
            self.fx_hist = self.fx_hist[-400:]
            self.fy_hist = self.fy_hist[-400:]
        x_min = min(self.fol_x, self.light_x) - 2.0
        x_max = max(self.fol_x, self.light_x) + 5.0
        cur_min, cur_max = self.ax.get_xlim()
        if x_min <= cur_min + 1 or x_max >= cur_max - 1:
            self.ax.set_xlim(x_min, x_max)
        self.fol_pt.set_data([self.fol_x], [0.0])
        self.fol_trail.set_data(self.fx_hist, self.fy_hist)
        self.light_pt.set_data([self.light_x], [0.0])
        self.light_pt.set_color('red' if self.light_red else 'green')
        gap_to_line = max(0.0, self.light_x - self.fol_x)
        state = 'RED' if self.light_red else 'GREEN'
        self.info.set_text(
            f"t={self.t:4.1f}s  light={state}  gap_to_line={gap_to_line:4.2f} m\n"
            f"v={self.fol_v:3.2f} m/s  v0={self.idm.v0:.2f}  T={self.idm.T:.2f}  s0={self.idm.s0:.2f}  a={self.idm.a_max:.2f}  b={self.idm.b_comf:.2f}  lim={self.sim.speed_limit:.2f}"
        )
        return self.fol_pt, self.fol_trail, self.light_pt, self.light_line, self.info
def main():
    idm = IDMParams()
    sim = SimParams()
    _ = IDMVisRedLight(idm, sim)
    plt.show()
if __name__ == "__main__":
    main()