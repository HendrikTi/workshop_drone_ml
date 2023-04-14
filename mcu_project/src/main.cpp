#include <Arduino.h>
#include "Wire.h"
#include <MPU6050_light.h>
#include <stdio.h>
#include <time.h>

#define TESTPIN 18

MPU6050 mpu(Wire);


volatile uint32_t isrCounter = 0;
volatile uint32_t lastIsrAt = 0;

long t_interv;

void setup()
{
  Serial.begin(115200);
  Wire.begin();

  byte status = mpu.begin();
  while (status != 0)
  {
    Serial.println(F("Error, could not connect to MPU6050!"));
  } // stop everything if could not connect to MPU6050

  delay(1000);
  mpu.calcOffsets(true, true); // gyro and accelero
  t_interv = millis();

}

void loop()
{
  
  uint8_t testpin_state = 0;
  char msg[100] = {0};
  float accx, accy, accz, gyrx, gyry, gyrz;
  int16_t cnt = 0;
  uint32_t phase = 1;
  while (1)
  {
    long t_now = millis();
    mpu.update();
    if (t_now > (t_interv + 100))
    {
      t_interv = millis();
      isrCounter++;
      digitalWrite(TESTPIN, testpin_state);
      testpin_state = !testpin_state;
      accx = mpu.getAccX();
      accy = mpu.getAccY();
      accz = mpu.getAccZ();
      gyrx = mpu.getGyroX();
      gyry = mpu.getGyroY();
      gyrz = mpu.getGyroZ();
      sprintf(msg, "%f,%f,%f,%f,%f,%f,%d", accx, accy, accz, gyrx, gyry, gyrz, phase);
      Serial.println(msg);
      if (cnt >= 30)
      {
        cnt = 0;
        phase++;
      }
      else
      {
        cnt++;
      }
    }
  }
}