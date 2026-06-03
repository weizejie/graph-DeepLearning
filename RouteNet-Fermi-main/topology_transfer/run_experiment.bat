@echo off
REM RouteNet-Fermi PyTorch - Topology Transfer Experiment
REM Train on fat128, then test on real traffic
REM Defaults match TF source: 5 epochs, 200 steps/epoch, MAPE loss

cd /d "%~dp0"

echo =============================================
echo RouteNet-Fermi PyTorch - Topology Transfer
echo =============================================

echo.
echo [Step 1] Training on fat128...
python train.py --train_dir ..\data\fat128\train --val_dir ..\data\fat128\test --epochs 5 --steps_per_epoch 200 --batch_size 1 --lr 0.001 --output_dir .\checkpoints --log_every 20 --val_steps 50

echo.
echo [Step 2] Testing on fat128 (same topology as training)...
python test.py --dataset fat128 --ckpt .\checkpoints\best.pt --max_samples 200

echo.
echo [Step 3] Testing on real traffic topologies (cross-topology generalization)...
python test.py --dataset real --ckpt .\checkpoints\best.pt --test_topo geant abilene nobel --max_samples 200

echo.
echo Done! Check results in .\results
pause
