# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: pysc2/env/converter/cc/game_data/proto/buffs.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n2pysc2/env/converter/cc/game_data/proto/buffs.proto\x12\x0bpysc2.Buffs*\x85\t\n\x05\x42uffs\x12\x0f\n\x0bUnknownBuff\x10\x00\x12\x10\n\x0c\x42\x61nsheeCloak\x10\x07\x12\x11\n\rBlindingCloud\x10S\x12\x1a\n\x16\x42lindingCloudStructure\x10&\x12%\n CarryHarvestableVespeneGeyserGas\x10\x91\x02\x12,\n\'CarryHarvestableVespeneGeyserGasProtoss\x10\x92\x02\x12)\n$CarryHarvestableVespeneGeyserGasZerg\x10\x93\x02\x12\'\n\"CarryHighYieldMineralFieldMinerals\x10\x90\x02\x12\x1e\n\x19\x43\x61rryMineralFieldMinerals\x10\x8f\x02\x12\x17\n\x12\x43hannelSnipeCombat\x10\x91\x01\x12\x0c\n\x08\x43harging\x10\x1e\x12\x1a\n\x15\x43hronoBoostEnergyCost\x10\x99\x02\x12\x14\n\x10\x43loakFieldEffect\x10\x1d\x12\x10\n\x0c\x43ontaminated\x10$\x12\x0e\n\nEMPDecloak\x10\x10\x12\x10\n\x0c\x46ungalGrowth\x10\x11\x12\x0e\n\nGhostCloak\x10\x06\x12\x11\n\rGhostHoldFire\x10\x0c\x12\x12\n\x0eGhostHoldFireB\x10\r\x12\x10\n\x0cGravitonBeam\x10\x05\x12\x12\n\x0eGuardianShield\x10\x12\x12\x14\n\x10ImmortalOverload\x10\x66\x12\x1f\n\x1aInhibitorZoneTemporalField\x10\xa1\x02\x12\n\n\x06LockOn\x10t\x12\x13\n\x0eLurkerHoldFire\x10\x88\x01\x12\x14\n\x0fLurkerHoldFireB\x10\x89\x01\x12\x15\n\x11MedivacSpeedBoost\x10Y\x12\x12\n\x0eNeuralParasite\x10\x16\x12\x14\n\x10OracleRevelation\x10\x31\x12\x1b\n\x16OracleStasisTrapTarget\x10\x81\x01\x12\x10\n\x0cOracleWeapon\x10\x63\x12\x12\n\rParasiticBomb\x10\x84\x01\x12%\n ParasiticBombSecondaryUnitSearch\x10\x86\x01\x12\x18\n\x13ParasiticBombUnitKU\x10\x85\x01\x12\x15\n\x11PowerUserWarpable\x10\x08\x12\x0c\n\x08PsiStorm\x10\x1c\x12\x18\n\x14QueenSpawnLarvaTimer\x10\x0b\x12\x1a\n\x15RavenScramblerMissile\x10\x95\x02\x12\'\n\"RavenShredderMissileArmorReduction\x10\x98\x02\x12\x1d\n\x18RavenShredderMissileTint\x10\x97\x02\x12\x08\n\x04Slow\x10!\x12\x0c\n\x08Stimpack\x10\x1b\x12\x14\n\x10StimpackMarauder\x10\x18\x12\x0e\n\nSupplyDrop\x10\x19\x12\x11\n\rTemporalField\x10y\x12\x19\n\x15ViperConsumeStructure\x10;\x12\x18\n\x13VoidRaySpeedUpgrade\x10\xa0\x02\x12\x1b\n\x17VoidRaySwarmDamageBoost\x10z')

_BUFFS = DESCRIPTOR.enum_types_by_name['Buffs']
Buffs = enum_type_wrapper.EnumTypeWrapper(_BUFFS)
UnknownBuff = 0
BansheeCloak = 7
BlindingCloud = 83
BlindingCloudStructure = 38
CarryHarvestableVespeneGeyserGas = 273
CarryHarvestableVespeneGeyserGasProtoss = 274
CarryHarvestableVespeneGeyserGasZerg = 275
CarryHighYieldMineralFieldMinerals = 272
CarryMineralFieldMinerals = 271
ChannelSnipeCombat = 145
Charging = 30
ChronoBoostEnergyCost = 281
CloakFieldEffect = 29
Contaminated = 36
EMPDecloak = 16
FungalGrowth = 17
GhostCloak = 6
GhostHoldFire = 12
GhostHoldFireB = 13
GravitonBeam = 5
GuardianShield = 18
ImmortalOverload = 102
InhibitorZoneTemporalField = 289
LockOn = 116
LurkerHoldFire = 136
LurkerHoldFireB = 137
MedivacSpeedBoost = 89
NeuralParasite = 22
OracleRevelation = 49
OracleStasisTrapTarget = 129
OracleWeapon = 99
ParasiticBomb = 132
ParasiticBombSecondaryUnitSearch = 134
ParasiticBombUnitKU = 133
PowerUserWarpable = 8
PsiStorm = 28
QueenSpawnLarvaTimer = 11
RavenScramblerMissile = 277
RavenShredderMissileArmorReduction = 280
RavenShredderMissileTint = 279
Slow = 33
Stimpack = 27
StimpackMarauder = 24
SupplyDrop = 25
TemporalField = 121
ViperConsumeStructure = 59
VoidRaySpeedUpgrade = 288
VoidRaySwarmDamageBoost = 122


if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _BUFFS._serialized_start=68
  _BUFFS._serialized_end=1225
# @@protoc_insertion_point(module_scope)
