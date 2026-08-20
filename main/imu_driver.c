/**
 * Onboard QMI8658 6-axis IMU (accelerometer + gyroscope) driver and simple
 * shake-gesture detector, used to let the BOOT-button-free "wave the board"
 * gestures switch 五言/七言 (Z-axis shake) and re-seed the theme (Y-axis
 * shake) while the infinite poem stream is running.
 *
 * Pin/address/register map (SDA=GPIO48, SCL=GPIO47, addr 0x6A/0x6B,
 * WHO_AM_I=0x00 expecting 0x05, CTRL1/2/3/7 init bytes, 12-byte burst read
 * from 0x35) comes from a community-documented teardown of this exact board
 * (github.com/mrRobot62/esp32-s3-touch-lcd-2, doc/qmi8658-imu.md) -- the
 * official Waveshare schematic PDF wasn't machine-readable at the time this
 * was written. imu_driver_init() probes WHO_AM_I before doing anything else
 * and returns false (logged, non-fatal) if the sensor doesn't answer, so a
 * wrong pin/address assumption fails loudly on the serial log instead of
 * silently mis-wiring the board.
 */
#include "imu_driver.h"
#include "driver/i2c_master.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <string.h>
#include <stdlib.h>

static const char *TAG = "IMU";

#define IMU_SDA_GPIO 48
#define IMU_SCL_GPIO 47
#define IMU_I2C_PORT I2C_NUM_0
#define IMU_I2C_FREQ_HZ 400000

#define QMI8658_ADDR_PRIMARY 0x6A
#define QMI8658_ADDR_SECONDARY 0x6B
#define QMI8658_REG_WHO_AM_I 0x00
#define QMI8658_REG_CTRL1 0x02
#define QMI8658_REG_CTRL2 0x03
#define QMI8658_REG_CTRL3 0x04
#define QMI8658_REG_CTRL7 0x08
#define QMI8658_REG_DATA 0x35
#define QMI8658_WHO_AM_I_EXPECTED 0x05

#define ACCEL_SCALE_G (1.0f / 4096.0f)

// Poll cadence and shake-detection tuning. Peak-to-peak accel swing over a
// short rolling window, on whichever axis dominates, past a g threshold,
// counts as a shake; a cooldown after firing avoids one physical shake
// producing several toggles.
#define POLL_INTERVAL_MS 50
#define WINDOW_SAMPLES 8 // ~400ms at 50ms/sample
#define SHAKE_THRESHOLD_G 1.4f
#define DOMINANCE_RATIO 1.8f
#define COOLDOWN_MS 1000

static i2c_master_bus_handle_t s_bus = NULL;
static i2c_master_dev_handle_t s_dev = NULL;
static bool s_available = false;

static volatile imu_gesture_t s_pending_gesture = IMU_GESTURE_NONE;

static esp_err_t qmi_write_reg(uint8_t reg, uint8_t val)
{
    uint8_t buf[2] = {reg, val};
    return i2c_master_transmit(s_dev, buf, sizeof(buf), 100);
}

static esp_err_t qmi_read_regs(uint8_t reg, uint8_t *out, size_t len)
{
    return i2c_master_transmit_receive(s_dev, &reg, 1, out, len, 100);
}

static bool qmi_probe_and_open(uint16_t addr)
{
    i2c_device_config_t dev_config = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = addr,
        .scl_speed_hz = IMU_I2C_FREQ_HZ,
    };
    if (i2c_master_bus_add_device(s_bus, &dev_config, &s_dev) != ESP_OK) {
        return false;
    }

    uint8_t who_am_i = 0;
    esp_err_t err = qmi_read_regs(QMI8658_REG_WHO_AM_I, &who_am_i, 1);
    if (err == ESP_OK && who_am_i == QMI8658_WHO_AM_I_EXPECTED) {
        ESP_LOGI(TAG, "QMI8658 found at 0x%02X (WHO_AM_I=0x%02X)", addr, who_am_i);
        return true;
    }

    ESP_LOGW(TAG, "No QMI8658 at 0x%02X (err=%s, WHO_AM_I=0x%02X)", addr, esp_err_to_name(err), who_am_i);
    i2c_master_bus_rm_device(s_dev);
    s_dev = NULL;
    return false;
}

static void imu_task(void *arg)
{
    // Z axis -> meter toggle (IMU_GESTURE_UPDOWN), Y axis -> theme re-seed
    // (IMU_GESTURE_LEFTRIGHT), per explicit direction from testing on the
    // real board -- not derived from the display rotation like the earlier
    // X/Y mapping was, so it's specified directly rather than reasoned out.
    float win_y[WINDOW_SAMPLES] = {0};
    float win_z[WINDOW_SAMPLES] = {0};
    int win_idx = 0;
    int win_count = 0;
    TickType_t last_trigger = 0;

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(POLL_INTERVAL_MS));

        uint8_t raw[12];
        if (qmi_read_regs(QMI8658_REG_DATA, raw, sizeof(raw)) != ESP_OK) {
            continue;
        }
        int16_t ay = (int16_t)(raw[2] | (raw[3] << 8));
        int16_t az = (int16_t)(raw[4] | (raw[5] << 8));
        float accel_y_g = ay * ACCEL_SCALE_G;
        float accel_z_g = az * ACCEL_SCALE_G;

        win_y[win_idx] = accel_y_g;
        win_z[win_idx] = accel_z_g;
        win_idx = (win_idx + 1) % WINDOW_SAMPLES;
        if (win_count < WINDOW_SAMPLES) win_count++;

        float min_y = win_y[0], max_y = win_y[0], min_z = win_z[0], max_z = win_z[0];
        for (int i = 1; i < win_count; i++) {
            if (win_y[i] < min_y) min_y = win_y[i];
            if (win_y[i] > max_y) max_y = win_y[i];
            if (win_z[i] < min_z) min_z = win_z[i];
            if (win_z[i] > max_z) max_z = win_z[i];
        }
        float swing_y = max_y - min_y;
        float swing_z = max_z - min_z;

        TickType_t now = xTaskGetTickCount();
        if (win_count < WINDOW_SAMPLES || (now - last_trigger) < pdMS_TO_TICKS(COOLDOWN_MS)) {
            continue;
        }

        imu_gesture_t gesture = IMU_GESTURE_NONE;
        if (swing_z > SHAKE_THRESHOLD_G && swing_z > swing_y * DOMINANCE_RATIO) {
            gesture = IMU_GESTURE_UPDOWN; // meter toggle
        } else if (swing_y > SHAKE_THRESHOLD_G && swing_y > swing_z * DOMINANCE_RATIO) {
            gesture = IMU_GESTURE_LEFTRIGHT; // theme re-seed
        }

        if (gesture != IMU_GESTURE_NONE) {
            s_pending_gesture = gesture;
            last_trigger = now;
            win_count = 0; // don't let the same motion re-trigger once the cooldown ends
            ESP_LOGI(TAG, "Shake detected: %s (swing_y=%.2fg swing_z=%.2fg)",
                     gesture == IMU_GESTURE_UPDOWN ? "meter (Z)" : "theme (Y)", swing_y, swing_z);
        }
    }
}

bool imu_driver_init(void)
{
    i2c_master_bus_config_t bus_config = {
        .i2c_port = IMU_I2C_PORT,
        .sda_io_num = IMU_SDA_GPIO,
        .scl_io_num = IMU_SCL_GPIO,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    if (i2c_new_master_bus(&bus_config, &s_bus) != ESP_OK) {
        ESP_LOGW(TAG, "Failed to init I2C bus on SDA=%d SCL=%d", IMU_SDA_GPIO, IMU_SCL_GPIO);
        return false;
    }

    if (!qmi_probe_and_open(QMI8658_ADDR_PRIMARY) && !qmi_probe_and_open(QMI8658_ADDR_SECONDARY)) {
        ESP_LOGW(TAG, "QMI8658 not found -- shake gestures disabled, BOOT button still works");
        i2c_del_master_bus(s_bus);
        s_bus = NULL;
        return false;
    }

    if (qmi_write_reg(QMI8658_REG_CTRL1, 0x60) != ESP_OK ||
        qmi_write_reg(QMI8658_REG_CTRL2, 0x23) != ESP_OK ||
        qmi_write_reg(QMI8658_REG_CTRL3, 0x43) != ESP_OK ||
        qmi_write_reg(QMI8658_REG_CTRL7, 0x03) != ESP_OK) {
        ESP_LOGW(TAG, "QMI8658 found but init write failed -- shake gestures disabled");
        return false;
    }

    s_available = true;
    xTaskCreatePinnedToCore(imu_task, "imu_task", 3072, NULL, 3, NULL, 0);
    ESP_LOGI(TAG, "QMI8658 shake-gesture detection running");
    return true;
}

imu_gesture_t imu_driver_poll_gesture(void)
{
    if (!s_available) return IMU_GESTURE_NONE;
    imu_gesture_t g = s_pending_gesture;
    s_pending_gesture = IMU_GESTURE_NONE;
    return g;
}
