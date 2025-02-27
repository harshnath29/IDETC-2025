// Pin definitions
const int THERMISTOR_PIN = A0;
const int RELAY_PIN = 2;

// Thermistor characteristics
const float BETA = 4800;
const float T0 = 298.15;    // 25°C in Kelvin
const float R0 = 100000;    // 100k nominal resistance at 25°C
const float R_DIVIDER = 10000;  // 10k voltage divider resistor

// Temperature control settings
const float TARGET_TEMP = 65;  // Target temperature in Celsius
const float TEMP_TOLERANCE = 2.0;  // Temperature window (±2°C)

void setup() {
  Serial.begin(9600);
  pinMode(THERMISTOR_PIN, INPUT);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);  // Start with relay off
  
  Serial.println("Heating pad control started");
  Serial.println("Target temperature: " + String(TARGET_TEMP) + "°C");
}

void loop() {
  int Vo = analogRead(THERMISTOR_PIN);
  float resistance = R_DIVIDER / ((1023.0/Vo) - 1.0);
  float temperature = (BETA * T0) / (BETA + (T0 * log(resistance/R0)));
  temperature -= 273.15;  // Convert to Celsius
  
  // Control heating
  if (temperature < (TARGET_TEMP - TEMP_TOLERANCE)) {
    digitalWrite(RELAY_PIN, HIGH);  // Turn on heater
  } else if (temperature > (TARGET_TEMP + TEMP_TOLERANCE)) {
    digitalWrite(RELAY_PIN, LOW);   // Turn off heater
  }
  
  // Print status
  Serial.print("ADC: ");
  Serial.print(Vo);
  Serial.print(" Temp: ");
  Serial.print(temperature);
  Serial.print("°C Heater: ");
  Serial.println(digitalRead(RELAY_PIN) ? "ON" : "OFF");
  
  delay(1000);
}
