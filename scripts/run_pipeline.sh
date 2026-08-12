#!/usr/bin/env bash
# Full experiment pipeline, in dependency order. Assumes the PPO checkpoint
# already exists (bahn ppo); everything else is regenerated from seeds.
set -e
BAHN="${BAHN:-bahn}"

echo "### tuning the hand-set constants on the training split"
$BAHN tune --n-maps 60 --batch 30

echo "### behaviour clone (poster encoding)"
$BAHN bc --steps 60000 --epochs 60
echo "### behaviour clone (raw-range encoding ablation)"
$BAHN bc --encoding raw --steps 60000 --epochs 60

echo "### regenerating the map suite"
$BAHN maps

echo "### evaluating every arm on the held-out split"
$BAHN eval --split test --batch 30

echo "### look-ahead ablation"
$BAHN ablate --n-maps 60 --batch 30

echo "### rendering RESULTS.md and figures"
$BAHN report >/dev/null
$BAHN figures
echo "### PIPELINE COMPLETE"
