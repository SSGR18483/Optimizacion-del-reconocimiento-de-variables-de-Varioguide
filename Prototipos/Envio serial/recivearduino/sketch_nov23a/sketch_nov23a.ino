/**
 *    __  ___    ________
 *   / / / / |  / / ____/
 *  / / / /| | / / / __
 * / /_/ / | |/ / /_/ /
 * \____/  |___/\____/
 *
 * Robotat bridge firmware
 */

/** Handles incoming JSON messages and transform
 *  them to CPX format then transmits via UART 
 *  to a Crazyflie
 */

// ==============================================
// LIBRERIAS
// ==============================================
#include <WiFi.h>
#include <ArduinoJson.h>

// ==============================================
// VARIABLES
// ==============================================

const char* ssid = "GaliciaReyes";
const char* password =  "J26S21P17S29";
const char* host = "192.168.56.1";
const uint16_t port = 80;
WiFiClient client;

float data[3];

StaticJsonDocument<256> doc;

// ==============================================
// PROTOTIPO DE FUNCIONES
// ==============================================
void startWifi(const char* ssid, const char* password);
void connectTCP(const char*host, const uint16_t port);

// ==============================================
// SETUP
// ==============================================
void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);

  startWifi(ssid, password);
  connectTCP(host, port);
}

// ==============================================
// LOOP
// ==============================================
void loop() {

  if (client.available() > 0){
    //read back one line from the server
    String line = client.readStringUntil('}');
    line = line + "}";

    DeserializationError err = deserializeJson(doc, line);
    if(err.code() == DeserializationError::Ok){
      data[0] = (float)doc["data"][0];
      data[1] = (float)doc["data"][1];
      data[2] = (float)doc["data"][2];
    }
  }
}

// ==============================================
// FUNCIONES
// ==============================================
void startWifi(const char* ssid, const char* password){
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi..");
  }
 
  Serial.println("Connected to the WiFi network");
  Serial.println(WiFi.localIP());
}

void connectTCP(const char*host, const uint16_t port){
  while (!client.connect(host, port)) {
    Serial.println("Connection failed.");
    Serial.println("Waiting 5 seconds before retrying...");
    delay(5000);
  }
}
