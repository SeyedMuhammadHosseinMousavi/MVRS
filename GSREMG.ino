// --- EMG Setup ---
int emgPin = A1;        // Analog pin for EMG sensor
int emgValue = 0;       // EMG sensor value

// --- GSR Setup ---
const int sensorPin = A0;   // Analog pin for GSR sensor

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  Serial.println("Initializing...");
  Serial.println("EMG and GSR sensors ready.");
}

void loop() {
  // --- Read EMG ---
  emgValue = analogRead(emgPin);    // Read EMG sensor value

  // --- Read GSR ---
  int gsrValue = analogRead(sensorPin);  // Read GSR sensor value

  // --- Print Results to Serial Monitor ---
  Serial.print("EMG: ");
  Serial.print(emgValue);
  Serial.print("  GSR: ");
  Serial.println(gsrValue);

  delay(200);  // Optional delay for readability
}
