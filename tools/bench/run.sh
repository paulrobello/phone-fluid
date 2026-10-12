#!/bin/bash
# Usage: tools/bench/run.sh [cell sizes, e.g. 2.6,2.1,1.7]
set -euo pipefail
cd "$(dirname "$0")/../.."
out=build/bench
mkdir -p "$out"
xcrun -sdk macosx metal -std=metal3.1 -O3 -ffast-math -c PhoneFluid/Shaders/Simulation.metal -o "$out/sim.air"
xcrun -sdk macosx metallib "$out/sim.air" -o "$out/sim.metallib"
xcrun swiftc -O -swift-version 6 -import-objc-header PhoneFluid/Shaders/ShaderTypes.h \
  PhoneFluid/Sim/FluidSimulation.swift PhoneFluid/Sim/TankGeometry.swift \
  PhoneFluid/Sim/FluidPreset.swift tools/bench/main.swift -o "$out/bench"
"$out/bench" "$out/sim.metallib" "${1:-2.6,2.1,1.7}"
