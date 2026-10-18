@echo off
title Sepsis CDSS - Clinical Edge Deployment
echo Starting Sepsis Clinical Decision Support Edge Deployment...
cd /d "%~dp0"
.venv\Scripts\python.exe deploy_clinical_edge.py
pause
