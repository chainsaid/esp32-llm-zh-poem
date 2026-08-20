#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    IMU_GESTURE_NONE = 0,
    IMU_GESTURE_UPDOWN,    // shake along the IMU's Z axis -- toggles 五言/七言
    IMU_GESTURE_LEFTRIGHT, // shake along the IMU's Y axis -- re-seeds the theme
} imu_gesture_t;

/**
 * @brief Starts I2C, probes the onboard QMI8658 6-axis IMU, and if found
 * spawns a background task that polls it and detects shake gestures.
 *
 * @return true if the IMU answered and gesture detection is running; false
 * if the probe failed (wrong board / sensor not present / wiring mismatch)
 * -- callers should treat that as non-fatal and just not get gesture input.
 */
bool imu_driver_init(void);

/**
 * @brief Returns the most recently detected gesture and clears it (each
 * gesture is reported exactly once). Returns IMU_GESTURE_NONE if nothing
 * new was detected, including when imu_driver_init() never found a sensor.
 */
imu_gesture_t imu_driver_poll_gesture(void);

#ifdef __cplusplus
}
#endif
