#!/usr/bin/env python
# Copyright (c) 2023-2024 Leberwurscht

import itertools
import multiprocessing

import numpy as np
import scipy.integrate
from scipy import constants
import matplotlib.pyplot as plt
import matplotlib.widgets

### functions
def integrate_spectral(nu,gdd):
  gdd = np.broadcast_to(gdd, nu.shape)
  gd = scipy.integrate.cumtrapz(gdd, 2*np.pi*nu.ravel(), initial=0)

  where = abs(nu-360e12)<30e12
  weights = np.exp(-(nu-360e12)**2/15e12**2)
  gd -= np.average(gd[where], weights=weights[where])

  return gd

def differentiate_spectral(nu,phase):
  phase = np.broadcast_to(phase, nu.shape)
  gd = np.gradient(phase, 2*np.pi*nu)
  return gd

def phase_to_gdd(nu,phase):
  return -differentiate_spectral(nu,differentiate_spectral(nu,phase))

def gdd_to_phase(nu,gdd):
  return -integrate_spectral(nu,integrate_spectral(nu,gdd))

def center(At):
  t0i = np.argmax(abs(At))
  At = np.roll(At, -t0i + At.size//2)
  return At

def ft(axis, data):
  return scipy.fft.fftshift(scipy.fft.fft(scipy.fft.ifftshift(data))) * (axis[1]-axis[0])

def ift(axis, data):
  return scipy.fft.fftshift(scipy.fft.ifft(scipy.fft.ifftshift(data))) * (axis[1]-axis[0]) * axis.size

### nu and t axis

n_points, t_spacing = 15000, 0.3e-15
t = ( np.arange(n_points, dtype=float)-n_points//2 ) * t_spacing

nu_spacing = 1/t_spacing/n_points
nu_env = ( np.arange(n_points, dtype=float)-n_points//2 ) * nu_spacing

nu_carrier = constants.c/830e-9
nu = nu_env + nu_carrier
wl = constants.c/nu

### resonator configuration

phase_sapphire = gdd_to_phase(nu, 240e-30)
gain_sapphire = 1.028 * 1.034 / 1.002**5 * np.exp(-((wl-820e-9)/110e-9)**8)
sapphire_thickness = 4.5e-3
gamma = 2.2e-6/1e-2

phase_prismarm = gdd_to_phase(nu, -520e-30)
loss_prismarm = np.broadcast_to(0, nu.shape)

phase_OCarm = gdd_to_phase(nu, 42.57e-30)
loss_OCarm = np.broadcast_to(3.5e-2, nu.shape)

M = np.exp(-(t/400e-15)**2) * np.exp(-(t/120e-15)**8) # initial temporal modulation
K = lambda A, A0, m, sigma: np.exp(-( (abs(A)-A0)/(sigma*A0) )**m) # SAM
modulator_relposition = 1.

SAM = True
modulation = True
synchronization = True
running = True

###

def crystal_step(Anu,dz):
  Anu = Anu * np.exp(1j*phase_sapphire*dz/sapphire_thickness) * np.sqrt(gain_sapphire)**(dz/sapphire_thickness)

  At = ift(nu_env, Anu)
  peakint = abs(At).max()**2

  At = At * np.exp(-1j*gamma*abs(At)**2 * dz)
  Anu = ft(t, At)
  return Anu, peakint

def roundtrip(nu, Anu_A, steps=10):
  Anus = []

  Anu = Anu_A

  # Ti:Sa crystal
  Anus.append(Anu)
  peakints1 = []
  for step in range(steps):
    Anu, peakint = crystal_step(Anu,sapphire_thickness/steps)
    peakints1.append(peakint)
    if step==steps//2: Anus.append(Anu)

  Anus.append(Anu)

  # prism arm
  Anu = Anu * np.exp(1j*phase_prismarm/2) * np.sqrt(1-loss_prismarm)**.5
  Anus.append(Anu)
  Anu = Anu * np.exp(1j*phase_prismarm/2) * np.sqrt(1-loss_prismarm)**.5

  # Ti:Sa crystal
  Anus.append(Anu)
  peakints2 = []
  for step in range(steps):
    Anu, peakint = crystal_step(Anu,sapphire_thickness/steps)
    peakints2.append(peakint)

    if step==int(np.round(steps*3/4.5))-1:
      Anus.append(Anu)

  # output coupler arm
  Anu = Anu * np.exp(1j*phase_OCarm*modulator_relposition) * np.sqrt(1-loss_OCarm)**modulator_relposition

  # modulation, SAM, sychronization
  if modulation or SAM or synchronization:
    At = ift(nu_env, Anu)

    if synchronization:
      At = At * M
  
    if modulation:
      At = At * M

    if SAM:
      At = K(At, abs(At).max(), 24, 0.94) * At

    Anu = ft(t, At)

  # rest of output coupler arm
  Anu = Anu * np.exp(1j*phase_OCarm*(1-modulator_relposition)) * np.sqrt(1-loss_OCarm)**(1-modulator_relposition)

  return Anu, Anus, np.array(peakints1), np.array(peakints2)

def plotter(plotter_pipe, controller_pipe):
  fig = plt.figure()

  rows,cols = 4,2

  commands = "sam_on sam_off M_on M_off syn_on syn_off gdd+ gdd- gdd2+ gdd2- gain+ gain- start stop init init_lp".split()
  btns = []
  gs = fig.add_gridspec(rows,3*cols)
  for j in range(cols):
    if len(commands)==0: break

    for i in range(rows):
      if len(commands)==0: break

      command = commands.pop(0)
      ax = fig.add_subplot(gs[i,3*j])
      btns.append( matplotlib.widgets.Button(ax, command) )
      btns[-1].on_clicked(lambda *args,command=command: controller_pipe.send(command))

      command = commands.pop(0)
      ax = fig.add_subplot(gs[i,3*j+1])
      btns.append( matplotlib.widgets.Button(ax, command) )
      btns[-1].on_clicked(lambda *args,command=command: controller_pipe.send(command))

  assert len(commands)==0

  fig = plt.figure(figsize=(16,9))

  def callback():
    im = plt.imread("sketch.png")

    while plotter_pipe.poll():
      print("received")
      command, *args = plotter_pipe.recv()
  
      if command=="terminate": return False
      elif command=="plot":
        fig.clear()

        iteration,Anus,peakints1,peakints2,pwrs,gain_sapphire, phase_sapphire, loss_prismarm, phase_prismarm, loss_OCarm, phase_OCarm = args
 
        where = (wl>650e-9)&(wl<1000e-9)

        ax0 = fig.add_subplot(111)
        ax0.imshow(im, aspect="auto")
        ax0.set_position([0,0,1,1])

        fig.suptitle("Iteration {}".format(iteration), weight="bold")

        ax = fig.add_subplot(111)
        ax.set_position([.055,.06, .15, .20])
        ax.axhline(1, color="#aaa")
        ax.plot(wl/1e-9, 1-loss_prismarm, 'r', label="P$_{1,2}$ arm")
        ax.plot(wl/1e-9,gain_sapphire**2, 'g', label="crystal")
        ax.plot(wl/1e-9,1-loss_OCarm, 'b', label="OC arm")
        gain_total = gain_sapphire**2*(1-loss_prismarm)*(1-loss_OCarm)
        ax.plot(wl/1e-9, gain_total, 'k', label="total")
        ax.set_xlim((650,1000))
        ax.set_ylim((0.8,1.2))
        #ax.set_yscale("log"); ax.set_ylim((0.8,1/.8))
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("transmission")
        ax.legend(loc=4)

        ax = fig.add_subplot(111)
        ax.set_position([.255,.06, .15, .20])
        ax.axhline(1, color="#aaa")
        ax.plot(wl[where]/1e-9, gain_total[where], 'k')
        ax.set_xlim((650,1000))
        ax.set_ylim((0,1.2))
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("total gain")
        ax.set_title("1c", loc="left", y=.85, weight="bold")

        ax = fig.add_subplot(111)
        ax.set_position([.555,.06, .15, .20])
        ax.plot(wl[where]/1e-9,gdd_to_phase(nu,phase_to_gdd(nu,np.unwrap(phase_prismarm)))[where], 'r')
        ax.plot(wl[where]/1e-9,gdd_to_phase(nu,phase_to_gdd(nu,np.unwrap(2*phase_sapphire)))[where], 'g')
        ax.plot(wl[where]/1e-9,gdd_to_phase(nu,phase_to_gdd(nu,np.unwrap(phase_OCarm)))[where], 'b')
        phase_total = phase_prismarm+2*phase_sapphire+phase_OCarm
        ax.plot(wl[where]/1e-9,gdd_to_phase(nu,phase_to_gdd(nu,np.unwrap(phase_total)))[where] , 'k')
        ax.set_xlim((650,1000))
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("phase (rad)")
        ax.set_ylim((-10,20))

        ax = fig.add_subplot(111)
        ax.set_position([.765,.06, .15, .20])
        ax.plot(wl[where]/1e-9, phase_to_gdd(nu,phase_total)[where]/1e-30)
        ax.set_xlim((650,1000))
        ax.set_ylim((-500,500))
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("GDD (fs$^2$)")
        ax.set_title("1b", loc="left", y=.85, weight="bold")

        ax = fig.add_subplot(111)
        ax.set_position([.724,.8, .17, .1])
        ax.plot(peakints1)
        ax.set_ylim((0, peakints1.max()*1.1))
        ax.set_ylabel("peak intensity")
        ax.set_xticks([0, peakints1.size/2, peakints2.size])
        ax.set_xticklabels(["A","B", "C"])
        ax.invert_xaxis()

        ax = fig.add_subplot(111)
        ax.set_position([.724,.65, .17, .1])
        ax.plot(peakints2)
        ax.set_ylim((0, peakints2.max()*1.1))
        ax.set_ylabel("peak intensity")
        ax.set_xticks([0, peakints2.size*3/4.5])
        ax.set_xticklabels(["E","F"])

        ax = fig.add_subplot(111)
        ax.set_position([.724,.40, .17, .15])
        ax.plot(pwrs)
        ax.set_ylim((0,pwrs.max()))
        ax.set_title("history")
        ax.set_xlabel("round-trip")
        ax.set_ylabel("intracavity power")

        Anu_max = 0.6e-10
        At_max = 1513*1.3

        positions = [
            ([.524,.74, .07, .09], [.524,.867, .07, .09], "5a"),
            ([.424,.74, .07, .09], [.424,.867, .07, .09], "5b"),
            ([.324,.74, .07, .09], [.324,.867, .07, .09], "5c"),
            ([.224,.74, .07, .09], [.224,.867, .07, .09], "5d"),
            ([.234,.50, .07, .09], [.134,.50, .07, .09], "5e"),
            ([.234,.37, .07, .09], [.134,.37, .07, .09], "5f"),
        ]

        gs = fig.add_gridspec(len(Anus)+1,2)
        for i,(Anu_,(pos_nu,pos_t,title)) in enumerate(zip(Anus,positions)):
          ax = fig.add_subplot(gs[i,0])
          ax.set_position(pos_nu)
          ax.plot(wl/1e-9, abs(Anu_)**2/Anu_max**2, 'k')
          ax.set_xlim((650,1000))
          ax.set_ylim((0, 1.2))
          #ax.set_title("{:.3e}".format(np.sum(abs(Anu_)**2)), y=.55)
          ax.set_title(title, loc="left", y=.65, weight="bold")
          ax.set_xticks([650,1000])
          ax.set_xticklabels(["       650 nm","1000 nm      "])
          ax.set_ylabel("int.", labelpad=-7)

          ax2 = ax.twinx()
          ax2.set_position(pos_nu)
          ph = np.unwrap(np.angle(Anu_))
          ph -= ph[nu.size//2]
          ax2.plot(wl[where]/1e-9, -ph[where]+10, 'b--')
          ax2.set_xlim((650,1000))
          ax2.set_ylim((0,30))
          ax2.set_yticks([0,30])
          ax2.set_yticklabels(["0","30"], color="b")
          ax2.set_ylabel("phase", color="b", labelpad=-15)
      
          At = ift(nu_env, Anu_)
          ax = fig.add_subplot(gs[i,1])
          ax.set_position(pos_t)
          ax.plot(t/1e-15, abs(At)**2/At_max**2, 'k')
          ax.set_xlim((-150,150))
          ax.set_ylim((0, 1.1))
          ax.set_xticks([-150,150])
          ax.set_xticklabels(["    -150 fs","150 fs    "])
          ax.set_ylabel("int.", labelpad=-7)

      else:
        raise Exception("unknown command")
  
    fig.canvas.draw()
    return True

  timer = fig.canvas.new_timer(interval=333)
  timer.add_callback(callback)
  timer.start()

  plt.show()

Anu = 4e-15 * np.exp(-(nu_env/26e12)**2) * np.exp(-1j*(nu_env/15e12)**2)
Anu_init = Anu.copy()

plot_pipe, plotter_pipe = multiprocessing.Pipe()
control_pipe, controller_pipe = multiprocessing.Pipe()
plotter_process = multiprocessing.Process(target=plotter, args=(plotter_pipe,controller_pipe), daemon=True)
plotter_process.start()
print("started")

pwrs = [0]*500
for i in itertools.count():
  Anu_, Anus, peakints1, peakints2 = roundtrip(nu, Anu)
  pwr = np.sum(abs(Anu_)**2)
  if running: Anu = Anu_

  pwrs = pwrs[1:]+[pwr]

  if i%10==0: print(i)

  if i%50==0:
    print("sending")
    plot_pipe.send(("plot", i,np.array(Anus), peakints1, peakints2, np.array(pwrs), gain_sapphire, phase_sapphire, loss_prismarm, phase_prismarm, loss_OCarm, phase_OCarm))
    print("sent")

    while control_pipe.poll():
      command = control_pipe.recv()

      if command=="sam_on": 
        SAM = True
      elif command=="sam_off": 
        SAM = False

      elif command=="M_on": 
        modulation = True
      elif command=="M_off": 
        modulation = False

      elif command=="syn_on": 
        synchronization = True
      elif command=="syn_off": 
        synchronization = False

      elif command=="gain+": 
        gain_sapphire *= 1.002
      elif command=="gain-": 
        gain_sapphire /= 1.002

      elif command=="gdd+": 
        phase_prismarm += gdd_to_phase(nu, 5e-30)
      elif command=="gdd-": 
        phase_prismarm += gdd_to_phase(nu, -5e-30)

      elif command=="gdd2+": 
        phase_OCarm += gdd_to_phase(nu, 5e-30)
      elif command=="gdd2-": 
        phase_OCarm -= gdd_to_phase(nu, 5e-30)

      elif command=="start": 
        running = True
      elif command=="stop": 
        running = False

      elif command=="init": 
        Anu = Anu_init
      elif command=="init_lp": 
        Anu = Anu_init * 0.01
      else:
        raise ValueError("unknown command {}".format(command))

      print("{} executed".format(command))

print("finished")
plotter_process.join()
