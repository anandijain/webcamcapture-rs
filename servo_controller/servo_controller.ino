#include <Servo.h>

Servo servo_x;  // create servo object to control a servo
Servo servo_y;  // create servo object to control a servo

int potpin = 0;  // analog pin used to connect the potentiometer
// int x,y;    // variable to read the value from the analog pin
float x,y;
int sx, sy;

void setup() {
  servo_x.attach(9);  // attaches the servo on pin 9 to the servo object
  servo_y.attach(10);  // attaches the servo on pin 9 to the servo object
  Serial.begin(9600);  // Start serial communication at 9600 baud.

}




bool parseStringToFloats(String data, float &x, float &y) {
  // First, trim the string to remove whitespace and newlines
  data.trim();

  // Check if the string starts with '[' and ends with ']'
  if (data.startsWith("[") && data.endsWith("]")) {
    // Remove the brackets
    data = data.substring(1, data.length() - 1);

    // Find the position of the comma
    int commaIndex = data.indexOf(',');
    if (commaIndex != -1) {
      // Extract the substring before and after the comma
      String xStr = data.substring(0, commaIndex);
      String yStr = data.substring(commaIndex + 1);

        // Convert these substrings to floats
      x = xStr.toFloat();
      y = yStr.toFloat();
      return true;
    }
  }
  return false; // Return false if the format is incorrect
}

void loop() {
  if (Serial.available() > 0) {
    // Read the incoming string until a newline is received
    String data = Serial.readStringUntil('\n');

    if (parseStringToFloats(data, x, y)) {
      Serial.print("Received x: ");
      Serial.print(x);
      Serial.print(", y: ");
      Serial.println(y);
      
      // Check if the received values are within the allowed range
      if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {
        // Scale the float values from 0.0-1.0 to 0-180 degrees directly
        sx = x * 180;  // directly scale the float to the range 0 to 180
        sy = y * 180;

        Serial.print("Mapped x: ");
        Serial.print(sx);
        Serial.print(", y: ");
        Serial.println(sy);

        // Write scaled values to servos
        servo_x.write(sx);
        servo_y.write(sy);
      } else {
        Serial.println("Values out of range, skipping...");
      }
    } else {
      Serial.println("Invalid data");
    }
  }
}
