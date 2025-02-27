const int THERMISTOR_PIN = A0;
const int RELAY_PIN = 2;

const float BETA = 4800;
const float T0 = 298.15;
const float R0 = 100000;
const float R_DIVIDER = 10000;

float TARGET_TEMP = 65.0;
const float TEMP_TOLERANCE = 2.0;

void setup() {
  Serial.begin(9600);
  pinMode(THERMISTOR_PIN, INPUT);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  
  Serial.println("Heating pad control started");
  Serial.print("Target temperature: ");
  Serial.print(TARGET_TEMP);
  Serial.println("°C");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    if (command.startsWith("SET_TEMP ")) {
      TARGET_TEMP = command.substring(9).toFloat();
      Serial.print("New target temperature: ");
      Serial.print(TARGET_TEMP);
      Serial.println("°C");
    }
  }

  int Vo = analogRead(THERMISTOR_PIN);
  float resistance = R_DIVIDER / ((1023.0/Vo) - 1.0);
  float temperature = (BETA * T0) / (BETA + (T0 * log(resistance/R0))) - 273.15;
  
  if (temperature < (TARGET_TEMP - TEMP_TOLERANCE)) {
    digitalWrite(RELAY_PIN, HIGH);
  } else if (temperature > (TARGET_TEMP + TEMP_TOLERANCE)) {
    digitalWrite(RELAY_PIN, LOW);
  }
  
  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.print("°C, Heater: ");
  Serial.println(digitalRead(RELAY_PIN) ? "ON" : "OFF");
  
  delay(1000);
}
