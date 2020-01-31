import RPi.GPIO as GPIO
import time
import sys
import Adafruit_DHT
import paho.mqtt.client as mqtt

PIR_PIN = 17
DHT_PIN = 15
DHT_SENSOR = Adafruit_DHT.DHT22

BROKER = "10.12.13.62"
TOPIC_TEMP ="RSP3/DHT22/Temp"
TOPIC_HUMI ="RSP3/DHT22/Humedad"
TOPIC_PIR = "RSP3/PIR/Presencia"

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)

while True:
    client = mqtt.Client()
    client.connect(BROKER,1883,60)
    
    humidity, temperature = Adafruit_DHT.read_retry(DHT_SENSOR,DHT_PIN)

    if humidity is not None:
        MENSAJE_HUMI = "{1:0.1f}".format(temperature,humidity)
        client.publish(TOPIC_HUMI, MENSAJE_HUMI) 
        print("Humidity={1:0.1f}".format(temperature,humidity))
        time.sleep(1) 
    else:
        print("Failed to get reading Humidity")

    if temperature is not None:
        MENSAJE_TEMP = "{0:0.1f}".format(temperature,humidity)
        client.publish(TOPIC_TEMP, MENSAJE_TEMP)
        print("Temperature={0:0.1f}".format(temperature,humidity))
        time.sleep(1)
    else:
        print("Failed to get reading Temperature")
        
    if GPIO.input(PIR_PIN) == True:
        print("Detectando")
        client.publish(TOPIC_PIR,"1")
    else:
        print("No Detectando")
        client.publish(TOPIC_PIR,"0")


