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

mqttc = mqtt.Client()

def on_connect(mqttc, obj, flags, rc):
    print("rc: " + str(rc))
    
def on_disconnect(mqttc, obj, flags, rc):
    mqttc.connect(BROKER, 1883, 60)
    print("reconnecting...")

def on_publish(mqttc, obj, mid):
    print("mid: " + str(mid))
    

mqttc.on_connect = on_connect
mqttc.on_disconnect = on_disconnect
mqttc.on_publish = on_publish

mqttc.connect(BROKER, 1883, 60)

mqttc.loop_start()
    
while True:
    humidity, temperature = Adafruit_DHT.read_retry(DHT_SENSOR, DHT_PIN)

    if humidity is not None:
        MENSAJE_HUMI = "{1:0.1f}".format(temperature, humidity)
        mqttc.publish(TOPIC_HUMI, MENSAJE_HUMI) 
        print("Humidity={1:0.1f}".format(temperature, humidity))
    else:
        print("Failed to get reading Humidity")

    if temperature is not None:
        MENSAJE_TEMP = "{0:0.1f}".format(temperature, humidity)
        mqttc.publish(TOPIC_TEMP, MENSAJE_TEMP)
        print("Temperature={0:0.1f}".format(temperature, humidity))
    else:
        print("Failed to get reading Temperature")
        
    if GPIO.input(PIR_PIN):
        print("Detectando")
        client.publish(TOPIC_PIR, "1")
    else:
        print("No Detectando")
        client.publish(TOPIC_PIR, "0")
        
    time.sleep(1)
