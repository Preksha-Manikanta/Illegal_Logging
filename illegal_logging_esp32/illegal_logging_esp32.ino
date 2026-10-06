/*
  Illegal Logging Detection - ESP32 Firmware
  ESP32 + MAX4466 microphone

  MAX4466 wiring will be done by the faculty:
    VCC -> ESP32 3.3V
    GND -> ESP32 GND
    OUT -> ESP32 GPIO34

  This firmware samples the microphone at 16 kHz and sends
  raw ADC samples through USB Serial for the computer-side
  Python/YAMNet processing.
*/

const int MIC_PIN = 34;
const int SAMPLE_RATE = 16000;
const unsigned long SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE;

void setup() {
  Serial.begin(115200);
  delay(1000);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  Serial.println("ILLEGAL LOGGING ESP32 STARTED");
  Serial.println("MAX4466 microphone input ready");
}

void loop() {
  static unsigned long nextSample = micros();

  if ((long)(micros() - nextSample) >= 0) {
    nextSample += SAMPLE_PERIOD_US;

    int sample = analogRead(MIC_PIN);
    Serial.println(sample);
  }
}
