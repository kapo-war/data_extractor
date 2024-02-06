#!/usr/bin/python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run SC2 to play a game or a replay."""

import getpass
import platform
import sys
import time
import json # added by skyview
from google.protobuf.json_format import MessageToDict # added by skview
from tqdm import tqdm # added by skyview
import numpy as np
from math import log10, ceil
import os
import torch

from absl import app

from absl import flags
from pysc2 import maps
from pysc2 import run_configs
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import renderer_human
from pysc2.lib import replay
from pysc2.lib import stopwatch

from s2clientprotocol import sc2api_pb2 as sc_pb

from pysc2.tests import utils

from pysc2.lib import features as F

import gc
import ray



FLAGS = flags.FLAGS
flags.DEFINE_string("outdir", "/root/code/outtest", "Outdir")
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_bool("realtime", False, "Whether to run in realtime mode.")
flags.DEFINE_bool("full_screen", False, "Whether to run full screen.")

flags.DEFINE_float("fps", 22.4, "Frames per second to run the game.")
flags.DEFINE_integer("step_mul", 1, "Game steps per observation.")
flags.DEFINE_bool("render_sync", False, "Turn on sync rendering.")
point_flag.DEFINE_point("feature_screen_size", "84",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "128",
                        "Resolution for minimap feature layers.")
flags.DEFINE_integer("feature_camera_width", 24,
                     "Width of the feature layer camera.")
point_flag.DEFINE_point("rgb_screen_size", "256,192",
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", "128",
                        "Resolution for rendered minimap.")
point_flag.DEFINE_point("window_size", "640,480",
                        "Screen size if not full screen.")
flags.DEFINE_string("video", None, "Path to render a video of observations.")

flags.DEFINE_integer("max_game_steps", 0, "Total game steps to run.")
flags.DEFINE_integer("max_episode_steps", 0, "Total game steps per episode.")

flags.DEFINE_string("user_name", getpass.getuser(),
                    "Name of the human player for replays.")
flags.DEFINE_enum("user_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "User's race.")
flags.DEFINE_enum("bot_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "AI race.")
flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
                  "Bot's strength.")
flags.DEFINE_enum("bot_build", "random", sc2_env.BotBuild._member_names_,  # pylint: disable=protected-access
                  "Bot's build strategy.")
flags.DEFINE_bool("disable_fog", False, "Disable fog of war.")
flags.DEFINE_integer("observed_player", 2, "Which player to observe.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_string("map", None, "Name of a map to use to play.")
flags.DEFINE_bool("battle_net_map", False, "Use the battle.net map version.")

flags.DEFINE_string("map_path", None, "Override the map for this replay.")
flags.DEFINE_string("replay", "/mount/jyj/replays/2019_WCS_Spring/307e3655dee84b0d93b14685baf68355.SC2Replay", "Name of a replay to show.")


num_cpus = 4
ray.init(num_cpus = num_cpus, ignore_reinit_error=True)

def encode_to_bits(arr:np.array, bits_per_elem:int, buffer_type:type=np.uint8()) -> bytes:
    """encode one_dimensional numpy array to compact bit info
       We need this since the bool type requires 1 byte per element, but we need only 1 bit
    Args:
        arr (np.array): _description_
        bits_per_elem (int): _description_
        buffer_type (type, optional): buffer type when compress. Defaults to np.uint8().

    Returns:
        bytes: byte info of compressed array
    """
    bits_per_buffer = 8*buffer_type.itemsize
    
    outarr = np.zeros(ceil(len(arr)/(bits_per_buffer*bits_per_elem)), dtype=buffer_type)

    for idx, element in enumerate(arr):
        outarr[int(idx/(bits_per_buffer*bits_per_elem))] += (element << (bits_per_buffer-1-idx%bits_per_buffer))
        
    byte_array = outarr.tobytes()
    
    return byte_array

def decode_to_numpy_bool(byte_array:bytes, n_items:int, buffer_type:type=np.uint8):
    """decode compact bit information to numpy array

    Args:
        byte_array (bytes): byte array to change
        bits_per_elem (int): 
        n_items (int): number of items that decoded array has
        buffer_type (type, optional): type of buffer, recommanded to be same with encode function. Defaults to np.uint8.

    Returns:
        _type_: 1 demensional numpy array
    """
    bits_per_elem = 1
    bits_per_buffer = 8*buffer_type.itemsize
    read_arr = np.frombuffer(byte_array, dtype=buffer_type)
    decoded_arr = np.zeros(n_items, dtype=bool)
    #print(read_arr)
    for idx in range(n_items):
        decoded_arr[idx] = (read_arr[int(idx/(bits_per_buffer*bits_per_elem))] >> ((bits_per_buffer-1) - (idx%bits_per_buffer)))%(2 << bits_per_elem)

    
    return decoded_arr


class OutManager:
  def __init__(self, info, player_id=1, root_path = "/root/code/outtest/preprocessed", world_size=(64, 64), save_type = 2):
    self.save_type = save_type
    self.root_path = root_path
    self.save_type = save_type
    self.world_size = world_size
    self.n_actions = 0
    # init 순서 중요!
    self.__init_meta(info, player_id)
    self.__init_world(world_size) # (FEATURE 종류, LOOP 번호, X, Y)
    self.__init_units() # (LOOP 번호, UNIT ID)
    self.__init_vectors() # (FEATURE 종류, LOOP 번호, SCALAR)'
    
    
    
    if save_type == 0: # per game(original)
      if not os.path.exists(os.path.join(root_path, "state")):
        os.makedirs(os.path.join(root_path, "state"), exist_ok=True)
      if not os.path.exists(os.path.join(root_path, "action")):
        os.makedirs(os.path.join(root_path, "action"), exist_ok=True)
      if not os.path.exists(os.path.join(root_path, "state/world")):
        os.makedirs(os.path.join(root_path, "state/world"), exist_ok=True)
      if not os.path.exists(os.path.join(root_path, "state/vector")):
        os.makedirs(os.path.join(root_path, "state/vector"), exist_ok=True)
      if not os.path.exists(os.path.join(root_path, "state/unit")):
        os.makedirs(os.path.join(root_path, "state/unit"), exist_ok=True)
    elif save_type == 1:
      pass
    elif save_type == 2:
      self.height_map = None
    else:
      assert False, f"save type should be either 0, 1 or 2, Your input is {save_type}"
    
  
  def add_loop(self, obs, loop, upgrades, units):  
    if (self.save_type == (0 or 1)):
      self.__add_world(obs)
      self.__add_units(units=units)
      self.__add_vectors(obs, loop, upgrades=upgrades)
      self.n_actions += 1
    elif (self.save_type == 2):
      
      outarr = self.__build_world(obs)
      outarr += self.__build_vector(obs, upgrades, loop)
      outarr += self.__build_units(units=units)
      
      with open(os.path.join(self.root_path, f"{str(loop).zfill(5)}.bin"), "wb") as f:
        f.write(outarr)
    else:
      assert False, "No save type avilable"
      
    
    
  def save_files(self, n_actions):
    if self.save_type == 0:
      self.__save_world(n_actions)
      self.__save_units()
      self.__save_vector()
    elif self.save_type == 1:
      n = ceil(log10(self.n_loops))
      # save height map
      with open(os.path.join(self.root_path, "height_map.bin"), "wb") as f:
        f.write(self.world["height_map"][0:self.world_size[0]*self.world_size[1]*8])
      # save others
      for i in range(self.n_actions):
        out_bin = self.__encode_state2bin(idx=i)
        with open(os.path.join(self.root_path, str(i).zfill(n)+".bin"), "wb") as f:
          f.write(out_bin)
    elif self.save_type == 2:
      self.save_details()
      return
    
    with open(os.path.join(self.root_path, "details.json"), "w") as j:
      outdict = {}
      outdict["world"] = {"bits_per_pixel": self.world_bits_per_pixel, "height_map": os.path.join(self.root_path, "height_map.bin"), "size": self.world_size}
      outdict["units"] = {"index_info": self.unit_keys}
      outdict["vector"] = {"index_info": self.vector_keys}
      json.dump(outdict, j, indent=2)  
  
  def save_details(self):
    # save height map
    with open(os.path.join(self.root_path, "height_map.bin"), "wb") as f:
      f.write(self.height_map)
    with open(os.path.join(self.root_path, "details.json"), "w") as j:
      outdict = {}
      outdict["world"] = {"bits_per_pixel": self.world_bits_per_pixel, "height_map": os.path.join(self.root_path, "height_map.bin"), "size": self.world_size}
      outdict["units"] = {"index_info": self.unit_keys}
      outdict["vector"] = {"index_info": self.vector_keys}
      json.dump(outdict, j, indent=2)  
    
  # def save_file(self, loop):
    
  #   # save others

  #   out_bin = self.__encode_state2bin()
  #   with open(os.path.join(self.root_path, str(loop).zfill(5)+".bin"), "wb") as f:
  #     f.write(out_bin)
  
    

  def __init_world(self, world_size):
    self.world = {"height_map":None, "visibility_map":None, "creep":None, "player_relative":None, "alerts":None,\
                        "pathable":None, "buildable":None, "camera":None}
    self.world_bits_per_pixel = {"height_map":None, "visibility_map":None, "creep":None, "player_relative":None, "alerts":None,\
                                  "pathable":None, "buildable":None, "camera":None}
    #pdb.set_trace()
    self.c = 0
  def __init_units(self):
    self.units = []
    self.unit_keys = None
  def __init_vectors(self):
    self.n_upgrades = 320
    self.vector_keys = ["player_id", "minerals", "vespene", "food_used", "food_cap", "food_workers", \
                        "food_army", "idle_worker_count", "army_count", "warp_gate_count", "larva_count", \
                        "game_loop", "unit_counts", "prev_delay", "upgrades"] # home race랑 away race는 dataloader에서 합하는걸로
    self.vector = np.zeros((self.n_loops, len(self.vector_keys)), dtype=np.int16)
    self.upgrades = np.zeros((self.n_loops, self.n_upgrades), dtype=bool)
    self.index = 0
  def __init_meta(self, info, player_id):  
    self.info = info
    self.n_loops = info.game_duration_loops
    if (len(info.player_info) != 2):
      assert False, f"Only twwo players are available, your data has {len(info.payer_info)} players"
    for player in info.player_info:
      if (player.player_info.player_id == player_id):
          self.home_race = player.player_info.race_actual
      else:
        self.away_race = player.player_info.race_actual
    with open(os.path.join(self.root_path, "meta.json"), "w") as f:
      json.dump(MessageToDict(info), f, indent=2)

  def __add_world(self, obs):
    feature_plane = obs.observation.feature_layer_data.minimap_renders
    for idx, key in enumerate(self.world.keys()):
      if (hasattr(feature_plane, key)):
        if (self.world[key] is None):
          self.world[key] = getattr(feature_plane, key).data
          self.world_bits_per_pixel[key] = getattr(feature_plane, key).bits_per_pixel

        elif (key == "height_map"):
          pass
        else:
          self.world[key] += getattr(feature_plane, key).data
  def __add_units(self, units):
    
    self.units.append(np.array(units, dtype=np.int16))
    
    if (self.unit_keys is None):
      self.unit_keys = units[0]._index_names[0]
    return
  def __add_vectors(self, obs, loop, upgrades):
    """
    for idx, key in enumerate(self.vector_keys):
      self.vectors[idx] = obs["playerCommon"][key]
      """
    
    vector_raw = obs.observation.player_common
    for idx, key in enumerate(self.vector_keys):
      if (hasattr(vector_raw, key)):
        self.vector[self.index, idx] = getattr(vector_raw, key)
      elif (key == "game_loop"):
        self.vector[self.index, idx] = loop
      elif (key == "unit_counts"):
        self.vector[self.index, idx] = len(obs.observation.raw_data.units)
      elif (key == "upgrades"):
        for upgrade_id in upgrades:
          assert upgrade_id>=0, f"upgrade ID should be [0, {self.n_upgrades-1}], your id is {upgrade_id}"
          assert upgrade_id<self.n_upgrades, f"upgrade ID should be [0, {self.n_upgrades-1}], your id is {upgrade_id}"
          self.upgrades[self.index, upgrade_id] = True
        
        if self.c != len(upgrades): 
          self.c = len(upgrades)
      else:
        pass
        #print(key, end=" ")
    self.index += 1 
    #print()

  def __save_world(self, n_actions):
    base_path = os.path.join(self.root_path, "state/world")
    for key in self.world.keys():
      with open(os.path.join(base_path, f"{key}.bin") , "wb") as f:
        if (self.world[key] is None):
          #print(key)
          pass
        else:
          f.write(self.world[key])
    with open(os.path.join(base_path, "bits_info.json"), "w") as f:
      self.world_bits_per_pixel["n_actions"] = n_actions
      json.dump(self.world_bits_per_pixel, f, indent=2)
  def __save_units(self):
    out_units = None
    out_dict = {"index": self.unit_keys, "mapping": []}
    for units in self.units:
      if out_units is None:
        out_units = units
      else:
        try:
          if (units.shape[1] != 46):
            assert False, f"Shape error! {units.shape}"
          out_units = np.concatenate((out_units, units), axis = 0)
        except:
          pass
          
       
      out_dict["mapping"].append(len(units))
        
    with open(os.path.join(self.root_path, "state/unit/unit_index.json"), "w") as f:
      json.dump(out_dict, f, indent=2)
    np.save(os.path.join(self.root_path, "state/unit/unit.npy"), out_units)
  def __save_vector(self):
    out_dict = {}
    # convert one-hot encoded upgrades array to bin
    byte_arr = encode_to_bits(self.upgrades.flatten(), 1)  
    out_dict["upgrades_array_shape"] = self.upgrades.shape
    
    torch.save(torch.tensor(self.vector), os.path.join(self.root_path, "state/vector/vector.pt"))
    #torch.save(torch.tensor(self.upgrades), os.path.join(self.root_path, "state/vector/upgrade.pt"))
    with open(os.path.join(self.root_path, "state/vector/upgrade.bin"), "wb") as f:
      f.write(byte_arr)
    with open(os.path.join(self.root_path, "state/vector/upgrade.json"), "w") as j:
      json.dump(out_dict, j, indent=2)

  def __encode_state2bin(self, idx):
    total_bin = None
    # world info flatten
    
    # except height map
    for world_feature in self.world.keys():
      n_bits = self.world_bits_per_pixel[world_feature] * self.world_size[0] * self.world_size[1]
      if world_feature == "height_map":
        continue
      elif (total_bin is None):
        total_bin = self.world[world_feature][idx*n_bits/8 : (idx+1)*n_bits/8]
      else:
        total_bin += self.world[world_feature][idx*n_bits/8 : (idx+1)*n_bits/8]
    
    # vector info flatten
    total_bin += self.vector[idx].tobytes() # little endian
    total_bin += encode_to_bits(self.upgrades[idx], 1) # 320 bits
    
    # unit info flatten
    total_bin += self.units[idx].tobytes()
    return total_bin


  def __build_world(self, obs):
    
    feature_plane = obs.observation.feature_layer_data.minimap_renders
    outarr = None
    for key in (self.world.keys()):
      if (hasattr(feature_plane, key)):
        if (self.world_bits_per_pixel[key] is None):
          self.world_bits_per_pixel[key] = getattr(feature_plane, key).bits_per_pixel
        if (key == "height_map"):
          if (self.height_map is None):
            self.height_map = getattr(feature_plane, key).data
          else:
            pass
        elif (outarr is None):
          outarr = getattr(feature_plane, key).data
        else:
          outarr += getattr(feature_plane, key).data
    return outarr
  def __build_vector(self, obs, upgrades, loop):
    vector_raw = obs.observation.player_common
    outvec = np.zeros(14, dtype=np.int16)
    outupgrades = np.zeros(320, dtype=bool)
    idx = 0
    for key in (self.vector_keys):
      if (hasattr(vector_raw, key)):
        outvec[idx] = getattr(vector_raw, key)
        idx += 1
      elif (key == "game_loop"):
        outvec[idx] = loop
        idx += 1
      elif (key == "unit_counts"):
        outvec[idx] = len(obs.observation.raw_data.units)
        idx += 1
      elif (key == "upgrades"):
        for upgrade_id in upgrades:
          assert upgrade_id>=0, f"upgrade ID should be [0, {self.n_upgrades-1}], your id is {upgrade_id}"
          assert upgrade_id<self.n_upgrades, f"upgrade ID should be [0, {self.n_upgrades-1}], your id is {upgrade_id}"
          outupgrades[upgrade_id] = True

      else:
        pass
    self.index += 1 
    
    return (outvec.tobytes()) + encode_to_bits(outupgrades, 1)
  def __build_units(self, units):
    
    outunits = np.array(units, dtype=np.int16)
    
    if (self.unit_keys is None):
      self.unit_keys = units[0]._index_names[0]
    return (outunits.tobytes())



DEBUG = 0    
      
    
 
def main(absl_flags, i, valid_action_loops):
  """Run SC2 to play a game or a replay."""
  #import pdb;pdb.set_trace()
  run_config = run_configs.get()
  #print("@@@@@@@@@@@@@@@@@@")
  interface = sc_pb.InterfaceOptions()
  interface.raw = True
  interface.raw_affects_selection = True
  interface.raw_crop_to_playable_area = True
  interface.score = True
  interface.show_cloaked = True
  interface.show_burrowed_shadows = True
  interface.show_placeholders = True
  if absl_flags["feature_screen_size"] and absl_flags["feature_minimap_size"]:
    interface.feature_layer.width = absl_flags["feature_camera_width"]
    absl_flags["feature_screen_size"].assign_to(interface.feature_layer.resolution)
    absl_flags["feature_minimap_size"].assign_to(
        interface.feature_layer.minimap_resolution)
    interface.feature_layer.crop_to_playable_area = True
    interface.feature_layer.allow_cheating_layers = True
  if absl_flags["render"] and absl_flags["rgb_screen_size"] and absl_flags["rgb_minimap_size"]:
    absl_flags["rgb_screen_size"].assign_to(interface.render.resolution)
    absl_flags["rgb_minimap_size"].assign_to(interface.render.minimap_resolution)



  replay_data = run_config.replay_data(absl_flags["replay"])
  start_replay = sc_pb.RequestStartReplay(
      replay_data=replay_data,
      options=interface,
      disable_fog=absl_flags["disable_fog"],
      observed_player_id=absl_flags["observed_player"])
  version = replay.get_replay_version(replay_data)
  #import pdb; pdb.set_trace()
  run_config = run_configs.get(version=version)  # Replace the run config.

  with run_config.start(
      full_screen=absl_flags["full_screen"],
      window_size=absl_flags["window_size"],
      raw=True,
      want_rgb=interface.HasField("render")) as controller:
  
    
    info = controller.replay_info(replay_data)
    
 
    state_manager = OutManager(info=info, player_id=1, world_size=(128, 128), root_path=absl_flags["outdir"])

    map_path = absl_flags["map_path"] or info.local_map_path
    if map_path:
      start_replay.map_data = run_config.map_data(map_path,
                                                  len(info.player_info))
    
    for i in info.player_info:
      
      player_info = MessageToDict(i.player_info)
      map_name = info.map_name
      
      if (player_info["playerId"] == absl_flags["observed_player"] and player_info["raceActual"] == "Protoss"): # and map_name == "Port Aleksander LE"
        VALID = True
      else:
        VALID = False
      VALID = True

    controller.start_replay(start_replay)

  
    feat = F.features_from_game_info(game_info=controller.game_info(),
                                                 use_feature_units=True, use_raw_units=True,
                                                 use_unit_counts=True, use_raw_actions=True,
                                                 show_cloaked=True, show_burrowed_shadows=True, 
                                                 show_placeholders=True) 
    

    try:
      n_actions = 0
      n_gc = 0
      gc_threshold = 60
 
      for i in tqdm(range(state_manager.n_loops), desc=absl_flags["replay"].split("/")[-1], position=1, leave = True):
        t_start = time.time()
        if (not VALID):
          break
        
        
        controller.step(absl_flags["step_mul"])
        obs = controller.observe()
       
        
        obs_tans = feat.transform_obs(obs)
        upgrades = obs_tans.upgrades
        units = obs_tans.feature_units

        if (i in valid_action_loops):
          n_actions += 1
          state_manager.add_loop(obs, i, upgrades = upgrades, units = units)

        
        if (obs.player_result):
          break
        
        if (1/(time.time() - t_start) < gc_threshold and n_gc <= 10):
          gc.collect()
          n_gc += 1
        elif (n_gc > 10 and gc_threshold > 30):
          n_gc = 0
          gc_threshold -= 10
          
      if (VALID):
        state_manager.save_files(n_actions)  
    except KeyboardInterrupt:
      pass
  return i
  


# def entry_point():  # Needed so setup.py scripts work.
#   app.run(main)

def scheduler(unused_argv):
  # base path infos
  competition_name = "2019_WCS_Spring"# "2019_HomeStory_Cup_XIX"
  replays_base_path = f"/mount/jyj/replays/{competition_name}"
  output_base_path = f"/mount/jyj/preprocessed/{competition_name}"
  
  # get replay hashes
  replays_list = [replay_file.split(".")[0] for replay_file in os.listdir(replays_base_path)]
  i = 0
  for match in tqdm(replays_list):
    for player in (1, 2):
      # set flags
      FLAGS.replay = os.path.join(replays_base_path, f"{match}.SC2Replay")
      FLAGS.outdir = os.path.join(output_base_path, match, f"player{player}/state")
      FLAGS.observed_player = player
      
      # make directory if not exist
      if (not os.path.exists(FLAGS.outdir)):
        os.makedirs(FLAGS.outdir, exist_ok=True)
      
      try:
        
        # get valid actions
        valid_action_loops = [int(loop.split(".")[0]) for loop in os.listdir(os.path.join(output_base_path, match, f"player{player}/action/selected"))]
        
        
        # change flag to dict for using ray(not implemented yet)
        absl_flags = flags.FLAGS.flag_values_dict()
        
        # if something in folder
        if (len(os.listdir(FLAGS.outdir)) > 1):
          continue
        
        # if meta.json exist -> check race is Protoss
        if (os.path.exists(os.path.join(FLAGS.outdir, "meta.json"))):
          with open(os.path.join(FLAGS.outdir, "meta.json"), "r") as j:
            meta_dict = json.load(j)
            if (meta_dict["playerInfo"][player-1]["playerInfo"]["raceActual"] != "Protoss"):
              continue
            
        # extract state info
        main(absl_flags, i, valid_action_loops)
        
      except Exception as e:
        print(e)
        
        #sys.exit(0)
        pass
      

if __name__ == "__main__":
  app.run(scheduler)

