#!/bin/bash

# Script para iniciar Iris
sudo systemctl restart systemd-timesyncd && timedatectl status 
bash ~/Desktop/InspectionApp/setup/start_iris.sh

# Esto es para iniciar manualmente
# Mas comentarios de prueba
echo "Proceso finalizado. Presiona Enter para cerrar esta ventana."
read