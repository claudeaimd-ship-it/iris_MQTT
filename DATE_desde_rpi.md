#1. Desde la terminal
sudo nano /etc/systemd/timesyncd.conf

# 2. Buscar la línea #NTP=, quitar el "#" y poner la IP de Domingo
NTP=192.168.45.100

# 3. ctrl+o para guardar, enter para aceptar y ctrl+x para salir

# 4. Aplicar cambios
sudo systemctl restart systemd-timesyncd

# 5. Verificar que aparezca la dirección
timedatectl timesync-status
