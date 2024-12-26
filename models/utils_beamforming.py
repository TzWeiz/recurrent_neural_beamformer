"""
Contains the microphones and noise sources relative locations to be utilized for simulation or computing beampatterns.

"""
import numpy as np 
# import gpuRIR
import math
# import pyroomacoustics as pra
from typing import *



def UMA8(center_pos, radius=0.045, plane="xy", n_mics=7, dtype="float64", rotate_deg=-60+13, rotate_mode="clock_wise"):
   """ Gives the cartesian positions of each microphone parallel to the plane provided
   
   Parameters
   ----------
   center_pos : tuple, list or of length 3
       Cartesian position of the center microphone
   radius : float
       The radius of each microphone from the center microphone
   plane: "xy", "xz", or "yz"
      The plane of the microphone array to be parallel to.
   n_mics: int
      The number of microphones in the microphone array. Includes the center microphone

   Returns
   -------
   ndarray of shape [n_mics, 3]
      The cartesian coordinates of each microphones
   """
   assert plane in ["xy", "yz", "xz"]

   mic_pos = np.zeros((n_mics, 3))
   
   x, y, z = center_pos
   rot_per_mic = 2 * math.pi / (n_mics-1)
   rotate = np.radians(rotate_deg)
   if rotate_mode=="counter_clock_wise":
      angles = np.arange(rotate, 2*math.pi + rotate -rot_per_mic*2, rot_per_mic)
   elif rotate_mode=="clock_wise":
      angles = np.arange(2*math.pi + rotate ,rotate -rot_per_mic, -rot_per_mic)
      
   else:
      raise Exception(f"Incorrect rotate mode given: {rotate_mode}. Please set counter_clock_wise or clock_wise")
   angles = angles[:n_mics-1]
   # print(np.degrees(angles))
   
   if plane =="xy":
      X = radius * np.cos(angles) + x
      Y = radius * np.sin(angles) + y
      Z = [z] * (n_mics-1)
   elif plane =="yz":
      Y = radius * np.cos(angles) + y
      Z = radius * np.sin(angles) + z
      X = [x] * (n_mics-1)
   elif plane =="xz":
      X = radius * np.cos(angles) + x
      Z = radius * np.sin(angles) + z
      Y = [y] * (n_mics-1)

   mic_pos[0] = center_pos
   mic_pos[1:,0] = X
   mic_pos[1:,1] = Y
   mic_pos[1:,2] = Z
   # print(mic_pos)
   return mic_pos

def UAV(center_pos, rot_deg=180+30, radius=0.785, plane="xy", n_rotors=6, dtype="float64" , rotate_mode="counter_clock_wise"):
   """ Gives the carterian positions of each rotor parallel to the plane provided.
   
   Parameters
   ----------
   center_pos : tuple, list or of length 3
       Cartesian position of the center of UAV
   rot_deg: float, degree
      The angle to rotate along the plane to the first rotor. The angle starts from the lowest dimension axis of the plane, ie if plane="xy", rotation start from x. 
   radius : float
       The radius of each rotor from the center position. [m]
   plane: "xy", "xz", or "yz"
      The plane of the microphone array to be parallel to.
   n_rotors: int
      The number of rotors equally distributed along the circle.
   Returns
   -------
   ndarray of shape [n_rotors, 3]
      The cartesian coordinates of each rotor position
   """
   assert plane in ["xy", "yz", "xz"]

   rotor_pos = np.zeros((n_rotors, 3), dtype)
   
   x, y, z = center_pos
   rot_per_rotor = 2 * math.pi / n_rotors
   rotate = np.radians(rot_deg)
   if rotate_mode == "counter_clock_wise":
      angles = np.arange(rotate, 2*math.pi-rot_per_rotor+rotate, -rot_per_rotor)
   elif rotate_mode=="clock_wise":
      angles = np.arange(2*math.pi+rotate, rotate-rot_per_rotor , +rot_per_rotor)
   else:
      raise Exception(f"Incorrect rotate mode given: {rotate_mode}. Please set counter_clock_wise or clock_wise")
   angles = angles[:n_rotors]
   # print(np.degrees(angles))
   assert len(angles) == n_rotors
   if plane =="xy":
      X = radius * np.cos(angles) + x
      Y = radius * np.sin(angles) + y
      Z = [z] * n_rotors
   elif plane =="yz":
      Y = radius * np.cos(angles) + y
      Z = radius * np.sin(angles) + z
      X = [x] * n_rotors
   elif plane =="xz":
      X = radius * np.cos(angles) + x
      Z = radius * np.sin(angles) + z
      Y = [y] * n_rotors

   # mic_pos[:,0] = center_pos
   rotor_pos[:,0] = X
   rotor_pos[:,1] = Y
   rotor_pos[:,2] = Z
   return rotor_pos

def simulateRIR_cpu(room_size, ref_coeff, pos_src, pos_rcv, mic_pattern, rt60, fs, att_diff, orV_rcv=None, att_max=60, max_length=6000):
   """Generates the room impulse response following example in https://github.com/DavidDiazGuerra/gpuRIR/blob/master/examples/example.py.

   Parameters
   ----------
   room_size : tuple or list 
      Size of the room [m]
   ref_coeff : tuple or list with length of 6
      Reflection coefficients of each wall.
   pos_src : ndarray of shape [n_src, 3]
      Cartesian coordinates of the source positions [m]
   pos_rcv : ndarray of shape [n_rcv, 3]
      Cartesian coordinates of the receiver positions [m]
   mic_pattern: str
      The microphone (receiver) polar pattern. "omni", "card".
   rt60 : float
      Time for the RIR to reach 60dB of attenuation [s]
   fs : int
      Sampling frequency [Hz]
   att_diff : float
      Attenuation when start using the diffuse reverberation model [dB]
   orV_rcv: nparray of shape [3]
      Orientation of the receivers as vectors pointing in the same direction. None for omnidirection.
   att_max : float
      Attenuation at the end of the simulation [dB]
   
   Returns
   --------
   :obj:`np.ndarray` of shape [n_src, n_rcv, room_impulse_length]
      The room impulse from using RIR
   """
   room_sz = room_size
   # Tmax = room_impulse_length/fs
   beta = ref_coeff
   
   n_src = pos_src.shape[0]  # Number of sources
   n_rcv = pos_rcv.shape[0] # Number of receivers
   
   
   # room = ()
   # import pyroomacoustics as pra
   # room_size = [120, 100, 21]
   # pos_src = [60, 51, 20]
   # from utils_beamforming import UMA8
   # pos_rcv = UMA8([2.4,0,0])
   # pos_rcv += np.array([60, 50, 0])
   # fs = 16000
   
   walls = ["north", "south", "west", "east", "floor", "ceiling"]
   if isinstance(ref_coeff,(float, int)):
      absorption = 1 - ref_coeff      
   else:
      absorption = 1 - np.array(ref_coeff)
      absorption = {wall: absorp for wall, absorp in zip(walls, absorption)}

   room = pra.ShoeBox(room_size,  fs=fs, absorption=absorption, max_order=17) 
   for src in pos_src:
      room.add_source(src)

   mic_array = pra.MicrophoneArray(np.c_[pos_rcv.T], fs)
   room.add_microphone_array(mic_array)
   room.compute_rir()
   rir = room.rir
   
   rir = np.stack([np.stack([room.rir[rcv][src][:max_length] for rcv in range(n_rcv)]) for src in range(n_src)])
         

   #to look at what this is afterwards
   # orV_rcv = np.matlib.repmat(np.array([0,1,0]), n_rcv, 1) # Vectors pointing in the same direction than the receivers
   # Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, rt60) 
   # nb_img = gpuRIR.t2n( Tdiff, room_sz )
   
   # mic_pattern = "omni" # Receiver polar pattern
   # Tmax = gpuRIR.att2t_SabineEstimator(att_max, rt60)

   # rir = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)
   # rir = room.rir # not sure shape tho
   return rir

def simulateRIR(room_size, ref_coeff, pos_src, pos_rcv, mic_pattern, rt60, fs, att_diff, orV_rcv=None, att_max=60):
   """Generates the room impulse response following example in https://github.com/DavidDiazGuerra/gpuRIR/blob/master/examples/example.py.

   Parameters
   ----------
   room_size : tuple or list 
      Size of the room [m]
   ref_coeff : tuple or list with length of 6
      Reflection coefficients of each wall.
   pos_src : ndarray of shape [n_src, 3]
      Cartesian coordinates of the source positions [m]
   pos_rcv : ndarray of shape [n_rcv, 3]
      Cartesian coordinates of the receiver positions [m]
   mic_pattern: str
      The microphone (receiver) polar pattern. "omni", "card".
   rt60 : float
      Time for the RIR to reach 60dB of attenuation [s]
   fs : int
      Sampling frequency [Hz]
   att_diff : float
      Attenuation when start using the diffuse reverberation model [dB]
   orV_rcv: nparray of shape [3]
      Orientation of the receivers as vectors pointing in the same direction. None for omnidirection.
   att_max : float
      Attenuation at the end of the simulation [dB]
   
   Returns
   --------
   :obj:`np.ndarray` of shape [n_src, n_rcv, room_impulse_length]
      The room impulse from using GPU RIR
   """
   room_sz = room_size
   # Tmax = room_impulse_length/fs
   beta = ref_coeff
   
   n_src = pos_src.shape[0]  # Number of sources
   n_rcv = pos_rcv.shape[0] # Number of receivers
   
   #to look at what this is afterwards
   # orV_rcv = np.matlib.repmat(np.array([0,1,0]), n_rcv, 1) # Vectors pointing in the same direction than the receivers
   Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, rt60) 
   nb_img = gpuRIR.t2n( Tdiff, room_sz )
   # 
   mic_pattern = "omni" # Receiver polar pattern
   Tmax = gpuRIR.att2t_SabineEstimator(att_max, rt60)

   rir = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)
   # for c in range(rir.shape[1]):
   #    rir[:,c] = rir[:,c] / np.max(np.abs(rir[:,c]))

   return rir

def cart2pol(pos_refs, pos_dests, dtype="float64"):
   r"""Converts cartesian coordinates to polar coordinates. 
   
   Parameters
   ----------
   pos_refs: ndarray of shape [n_ref, 3]
      Cartesian coordinates of reference positions
   pos_dests: ndarray of shape [n_dest, 3]
      Cartesian coordinates of destinations
   Returns
   --------
   ndarray of shape [n_dest, n_ref, 3]
      Polar coordianates (azimuth, elev, height) from each reference to destination positions
      Azimuth in radians, Elevation in radians and Height
   """

   n_ref = pos_refs.shape[0]
   n_dest = pos_dests.shape[0]
   polar_coord = np.zeros((n_dest, n_ref, 3), dtype)

   x_d, y_d, z_d = pos_dests[:,0], pos_dests[:,1], pos_dests[:, 2]
   for i, (x, y, z) in enumerate(pos_refs):
      azimuth =  np.arctan2(y_d - y, x_d - x)
      rho = np.sqrt( (x_d - x) ** 2 + (y_d - y) ** 2 + (z_d - z) **2 )
      height = z_d - z
      elevation = np.arcsin(height/rho)
      polar_coord[:, i, 0] = azimuth
      polar_coord[:, i, 1] = elevation
      polar_coord[:, i, 2] = height
   
   return polar_coord


def steering_vector(freq, pos_refs, pos_lkdirs, c=343, dtype="complex128"):
   r"""Computes the steering vector. 

   #todo: add equation
   
   Parameters
   ----------
   freq : ndarray
      The frequency range. [Hz]
   pos_refs : ndarray of shape [n_ref, 3]
      Cartesian coordinates of reference positions. [m]
   pos_lkdirs: ndarray of shape [n_src, 3]
      Cartesian coordinates of look directions. [m]
   c: float or int
      The speed of sound. [m/s]

   Returns
   --------
   ndarray of shape [n_freq, n_lkdir, n_ref]
   The steering vector of 
   """
   n_lkdir = pos_lkdirs.shape[0]
   n_ref = pos_refs.shape[0]
   n_freq = len(freq)
   steering_vectors = np.zeros((n_freq, n_ref, n_lkdir),dtype)
   polar_coords = cart2pol(pos_refs, pos_lkdirs) #: shape of [n_lkdirs, n_ref, 3]

   freq = np.expand_dims(freq, axis=-1)
   
   # for i in range(n_lkdir):
   for i in range(n_ref):
      azimuth, elevation, _ = polar_coords[:,i,0], polar_coords[:,i,1], polar_coords[:,i,2]
      ref_x, ref_y, ref_z = pos_refs[i]
      v = np.exp(2j * np.pi * freq/ c *
                  (ref_x * np.cos(azimuth) * np.cos(elevation) + \
                  ref_y * np.sin(azimuth) * np.cos(elevation) + \
                  ref_z * np.sin(elevation)))
      steering_vectors[:,i,:] = v
   return steering_vectors

