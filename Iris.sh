#!/bin/bash

# Script para iniciar Iris
sudo systemctl restart systemd-timesyncd && timedatectl status 
bash ~/Desktop/InspectionApp/setup/start_iris.sh

# Esto es para iniciar manualmente
# Y esto para iniciar automáticamente al encender la computadora
echo "Proceso finalizado. Presiona Enter para cerrar esta ventana."
read