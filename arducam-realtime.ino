#include <Wire.h>
#include <BreezyArduCAM.h>
#include <SPI.h>

static const int CS = 13;  // Chip select pin for the ArduCAM

Serial_ArduCAM_FrameGrabber fg;

// Choose your camera model: 2MP or 5MP
ArduCAM_Mini_2MP myCam(CS, &fg);
// ArduCAM_Mini_5MP myCam(CS, &fg);

void setup(void)
{
    // Initialize I2C and SPI communication
    Wire.begin();
    SPI.begin();

    // Initialize serial communication
    Serial.begin(921600); // Baud rate to match with Python code

    // Start the camera in JPEG mode (320x240 resolution)
    myCam.beginJpeg320x240();
}

void loop(void)
{
    // Capture a JPEG image
    myCam.capture();
   
    // Read and send the data from the FIFO buffer in small chunks
    while (true) {
        uint8_t byte = myCam.read_fifo();  // Read a single byte from the FIFO buffer
        if (byte == 0xFF) {  // Check if it's the end of the JPEG (this is just a sample check)
            Serial.write(byte);  // Write the byte to the serial port
            break;  // Exit the loop once we reach the end marker
        }
        Serial.write(byte);  // Write the byte to the serial port
    }
}
