#!/usr/bin/env bash

get_can0_status() {
        if [ "`ifconfig | grep can0`" == "" ]; then
                echo "closed"
        else
                echo "open"
        fi
}

[ `id -u` -ne 0 ] && echo "run this script as sudo." && exit 0

[ "`sudo dmesg | grep mcp`" == "" ] && echo "MCP was not initialized succ
esfully" && exit 1

cur_status="$(get_can0_status)"

if [ "$1" == "loop" ] && [ $cur_status == "closed" ]; then
        mode="up type can bitrate 500000 loopback on"
elif [ "$1"  == "close" ] && [ $cur_status == "open" ]; then
        mode="down"
elif [ "$1" == "open" ] && [ $cur_status == "closed" ]; then
  mode="up type can bitrate 500000"
elif [ $cur_status == "closed" ]; then
  cat << EOS 
  usage:
    sudo ./open-bus.sh open    # open can0 without loopback
    sudo ./open-bus.sh close   # close can0
    sudo ./open-bus.sh loop    # open can0 with loopback
EOS
        exit 0
fi

sudo ip link set can0 $mode
open-bus.sh
echo "can0 is: $(get_can0_status)"
