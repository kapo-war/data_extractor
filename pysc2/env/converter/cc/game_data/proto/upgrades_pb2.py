# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: pysc2/env/converter/cc/game_data/proto/upgrades.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n5pysc2/env/converter/cc/game_data/proto/upgrades.proto\x12\x0epysc2.Upgrades*\xe4\x11\n\x08Upgrades\x12\x13\n\x0e\x41\x64\x61ptiveTalons\x10\xa5\x02\x12\x11\n\rAdrenalGlands\x10\x41\x12\x17\n\x12\x41\x64vancedBallistics\x10\x8c\x01\x12\x15\n\x11\x41nabolicSynthesis\x10X\x12\x16\n\x12\x41nionPulseCrystals\x10\x63\x12\t\n\x05\x42link\x10W\x12\n\n\x06\x42urrow\x10@\x12\x14\n\x10\x43\x65ntrificalHooks\x10K\x12\n\n\x06\x43harge\x10V\x12\x14\n\x10\x43hitinousPlating\x10\x04\x12\x11\n\rCloakingField\x10\x14\x12\x10\n\x0c\x43ombatShield\x10\x10\x12\x14\n\x10\x43oncussiveShells\x10\x11\x12\x11\n\rCorvidReactor\x10\x16\x12\x1e\n\x19\x43ycloneRapidFireLaunchers\x10\xa3\x02\x12\x11\n\rDrillingClaws\x10z\x12\x17\n\x12\x45nhancedShockwaves\x10\xa8\x02\x12\x18\n\x14\x45xtendedThermalLance\x10\x32\x12\x17\n\x13GlialReconstitution\x10\x02\x12\x13\n\x0fGraviticBooster\x10\x30\x12\x11\n\rGraviticDrive\x10\x31\x12\x14\n\x10GravitonCatapult\x10\x01\x12\x12\n\rGroovedSpines\x10\x86\x01\x12\x15\n\x11HiSecAutoTracking\x10\x05\x12\x1a\n\x15HighCapacityFuelTanks\x10\x8b\x01\x12\x16\n\x11HyperflightRotors\x10\x88\x01\x12\x16\n\x12InfernalPreigniter\x10\x13\x12\x0b\n\x06LockOn\x10\x90\x01\x12\x12\n\x0eMetabolicBoost\x10\x42\x12\x15\n\x10MuscularAugments\x10\x87\x01\x12\x11\n\rNeosteelFrame\x10\n\x12\x12\n\x0eNeuralParasite\x10\x65\x12\x12\n\x0ePathogenGlands\x10J\x12\x14\n\x10PersonalCloaking\x10\x19\x12\x17\n\x13PneumatizedCarapace\x10>\x12\x1a\n\x16ProtossAirArmorsLevel1\x10Q\x12\x1a\n\x16ProtossAirArmorsLevel2\x10R\x12\x1a\n\x16ProtossAirArmorsLevel3\x10S\x12\x1b\n\x17ProtossAirWeaponsLevel1\x10N\x12\x1b\n\x17ProtossAirWeaponsLevel2\x10O\x12\x1b\n\x17ProtossAirWeaponsLevel3\x10P\x12\x1d\n\x19ProtossGroundArmorsLevel1\x10*\x12\x1d\n\x19ProtossGroundArmorsLevel2\x10+\x12\x1d\n\x19ProtossGroundArmorsLevel3\x10,\x12\x1e\n\x1aProtossGroundWeaponsLevel1\x10\'\x12\x1e\n\x1aProtossGroundWeaponsLevel2\x10(\x12\x1e\n\x1aProtossGroundWeaponsLevel3\x10)\x12\x18\n\x14ProtossShieldsLevel1\x10-\x12\x18\n\x14ProtossShieldsLevel2\x10.\x12\x18\n\x14ProtossShieldsLevel3\x10/\x12\x0c\n\x08PsiStorm\x10\x34\x12\x16\n\x11ResonatingGlaives\x10\x82\x01\x12\x11\n\x0cShadowStrike\x10\x8d\x01\x12\x10\n\x0bSmartServos\x10\xa1\x02\x12\x0c\n\x08Stimpack\x10\x0f\x12\x1e\n\x1aTerranInfantryArmorsLevel1\x10\x0b\x12\x1e\n\x1aTerranInfantryArmorsLevel2\x10\x0c\x12\x1e\n\x1aTerranInfantryArmorsLevel3\x10\r\x12\x1f\n\x1bTerranInfantryWeaponsLevel1\x10\x07\x12\x1f\n\x1bTerranInfantryWeaponsLevel2\x10\x08\x12\x1f\n\x1bTerranInfantryWeaponsLevel3\x10\t\x12\x1b\n\x17TerranShipWeaponsLevel1\x10$\x12\x1b\n\x17TerranShipWeaponsLevel2\x10%\x12\x1b\n\x17TerranShipWeaponsLevel3\x10&\x12\x18\n\x14TerranStructureArmor\x10\x06\x12$\n TerranVehicleAndShipArmorsLevel1\x10t\x12$\n TerranVehicleAndShipArmorsLevel2\x10u\x12$\n TerranVehicleAndShipArmorsLevel3\x10v\x12\x1e\n\x1aTerranVehicleWeaponsLevel1\x10\x1e\x12\x1e\n\x1aTerranVehicleWeaponsLevel2\x10\x1f\x12\x1e\n\x1aTerranVehicleWeaponsLevel3\x10 \x12\x12\n\x0eTunnelingClaws\x10\x03\x12\x14\n\x10WarpGateResearch\x10T\x12\x0f\n\x0bWeaponRefit\x10L\x12\x19\n\x15ZergFlyerArmorsLevel1\x10G\x12\x19\n\x15ZergFlyerArmorsLevel2\x10H\x12\x19\n\x15ZergFlyerArmorsLevel3\x10I\x12\x1a\n\x16ZergFlyerWeaponsLevel1\x10\x44\x12\x1a\n\x16ZergFlyerWeaponsLevel2\x10\x45\x12\x1a\n\x16ZergFlyerWeaponsLevel3\x10\x46\x12\x1a\n\x16ZergGroundArmorsLevel1\x10\x38\x12\x1a\n\x16ZergGroundArmorsLevel2\x10\x39\x12\x1a\n\x16ZergGroundArmorsLevel3\x10:\x12\x1a\n\x16ZergMeleeWeaponsLevel1\x10\x35\x12\x1a\n\x16ZergMeleeWeaponsLevel2\x10\x36\x12\x1a\n\x16ZergMeleeWeaponsLevel3\x10\x37\x12\x1c\n\x18ZergMissileWeaponsLevel1\x10;\x12\x1c\n\x18ZergMissileWeaponsLevel2\x10<\x12\x1c\n\x18ZergMissileWeaponsLevel3\x10=')

_UPGRADES = DESCRIPTOR.enum_types_by_name['Upgrades']
Upgrades = enum_type_wrapper.EnumTypeWrapper(_UPGRADES)
AdaptiveTalons = 293
AdrenalGlands = 65
AdvancedBallistics = 140
AnabolicSynthesis = 88
AnionPulseCrystals = 99
Blink = 87
Burrow = 64
CentrificalHooks = 75
Charge = 86
ChitinousPlating = 4
CloakingField = 20
CombatShield = 16
ConcussiveShells = 17
CorvidReactor = 22
CycloneRapidFireLaunchers = 291
DrillingClaws = 122
EnhancedShockwaves = 296
ExtendedThermalLance = 50
GlialReconstitution = 2
GraviticBooster = 48
GraviticDrive = 49
GravitonCatapult = 1
GroovedSpines = 134
HiSecAutoTracking = 5
HighCapacityFuelTanks = 139
HyperflightRotors = 136
InfernalPreigniter = 19
LockOn = 144
MetabolicBoost = 66
MuscularAugments = 135
NeosteelFrame = 10
NeuralParasite = 101
PathogenGlands = 74
PersonalCloaking = 25
PneumatizedCarapace = 62
ProtossAirArmorsLevel1 = 81
ProtossAirArmorsLevel2 = 82
ProtossAirArmorsLevel3 = 83
ProtossAirWeaponsLevel1 = 78
ProtossAirWeaponsLevel2 = 79
ProtossAirWeaponsLevel3 = 80
ProtossGroundArmorsLevel1 = 42
ProtossGroundArmorsLevel2 = 43
ProtossGroundArmorsLevel3 = 44
ProtossGroundWeaponsLevel1 = 39
ProtossGroundWeaponsLevel2 = 40
ProtossGroundWeaponsLevel3 = 41
ProtossShieldsLevel1 = 45
ProtossShieldsLevel2 = 46
ProtossShieldsLevel3 = 47
PsiStorm = 52
ResonatingGlaives = 130
ShadowStrike = 141
SmartServos = 289
Stimpack = 15
TerranInfantryArmorsLevel1 = 11
TerranInfantryArmorsLevel2 = 12
TerranInfantryArmorsLevel3 = 13
TerranInfantryWeaponsLevel1 = 7
TerranInfantryWeaponsLevel2 = 8
TerranInfantryWeaponsLevel3 = 9
TerranShipWeaponsLevel1 = 36
TerranShipWeaponsLevel2 = 37
TerranShipWeaponsLevel3 = 38
TerranStructureArmor = 6
TerranVehicleAndShipArmorsLevel1 = 116
TerranVehicleAndShipArmorsLevel2 = 117
TerranVehicleAndShipArmorsLevel3 = 118
TerranVehicleWeaponsLevel1 = 30
TerranVehicleWeaponsLevel2 = 31
TerranVehicleWeaponsLevel3 = 32
TunnelingClaws = 3
WarpGateResearch = 84
WeaponRefit = 76
ZergFlyerArmorsLevel1 = 71
ZergFlyerArmorsLevel2 = 72
ZergFlyerArmorsLevel3 = 73
ZergFlyerWeaponsLevel1 = 68
ZergFlyerWeaponsLevel2 = 69
ZergFlyerWeaponsLevel3 = 70
ZergGroundArmorsLevel1 = 56
ZergGroundArmorsLevel2 = 57
ZergGroundArmorsLevel3 = 58
ZergMeleeWeaponsLevel1 = 53
ZergMeleeWeaponsLevel2 = 54
ZergMeleeWeaponsLevel3 = 55
ZergMissileWeaponsLevel1 = 59
ZergMissileWeaponsLevel2 = 60
ZergMissileWeaponsLevel3 = 61


if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _UPGRADES._serialized_start=74
  _UPGRADES._serialized_end=2350
# @@protoc_insertion_point(module_scope)
