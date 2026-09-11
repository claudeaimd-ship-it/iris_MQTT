# 1. Configurar la IP fija, el Gateway y el DNS
sudo nmcli connection modify "JE-DA&A" ipv4.method manual ipv4.addresses "192.168.45.[IP_NUEVA]/24" ipv4.gateway "192.168.45.1" ipv4.dns "192.168.45.1"

# 2. Agregar la ruta estática hacia el servidor Windows
sudo nmcli connection modify "JE-DA&A" +ipv4.routes "172.24.217.0/24 192.168.45.1"

# 3. Reiniciar la interfaz para aplicar los cambios instantáneamente
sudo nmcli connection up "JE-DA&A"

# ─────────────────────────────────────────────────────────────────────────
# IP fija en eth0 para IrisLink (cable Ethernet directo PC↔Pi)
# ─────────────────────────────────────────────────────────────────────────
# El puerto Ethernet físico de la Pi no se usa para la red corporativa (esa
# va por WiFi, ver arriba) — queda libre para el cable directo de IrisLink.
# Usa el MISMO último octeto que la IP de WiFi (192.168.45.[IP_NUEVA]) para
# poder identificar la Pi a simple vista desde este otro puerto/subred.

# 4. Crear la conexión de eth0 con IP fija en la subred dedicada 10.55.0.0/24
sudo nmcli connection add type ethernet ifname eth0 con-name "IrisLink-eth0" \
  ipv4.method manual ipv4.addresses "10.55.0.[IP_NUEVA]/24"
sudo nmcli connection up "IrisLink-eth0"

# 5. Crear /etc/dnsmasq.d/irislink.conf con su contenido (copiar/pegar tal cual)
#    NOTA 1: se usa "bind-dynamic" en vez de "bind-interfaces" — con
#    "bind-interfaces" dnsmasq FALLA al bootear si eth0 todavía no existe/
#    está lista en ese instante (carrera de arranque con NetworkManager,
#    error típico: "dnsmasq: interfase desconocida eth0"). "bind-dynamic"
#    tolera que la interfaz aparezca después.
#    NOTA 2: "dhcp-option=3,..." (router/gateway) es OBLIGATORIO para que
#    IrisLink pueda "Auto-detectar" la Pi — sin esta opción, el cliente
#    DHCP de la PC no recibe gateway y `detect_pi_ip()` (que busca la Pi
#    leyendo el gateway por defecto de la interfaz) nunca la encuentra,
#    aunque la PC sí reciba IP y la conexión funcione con "IP manual".
#    IMPORTANTE: reemplazar [IP_NUEVA] por el mismo octeto usado en el
#    paso 4 (la propia IP de esta Pi en la subred 10.55.0.0/24).
sudo tee /etc/dnsmasq.d/irislink.conf > /dev/null << 'EOF'
interface=eth0
bind-dynamic
dhcp-range=10.55.0.100,10.55.0.150,12h
dhcp-option=3,10.55.0.[IP_NUEVA]
EOF

sudo systemctl restart dnsmasq
sudo systemctl enable dnsmasq

# Con esto, al conectar el cable, la PC (IrisLink) recibe IP automáticamente
# por DHCP sin necesitar configuración manual de red, y detecta la Pi leyendo
# el gateway por defecto de esa interfaz (que es la IP fija de eth0 de arriba).

