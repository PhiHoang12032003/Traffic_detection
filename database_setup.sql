-- Traffic Monitoring System Database Setup
-- MySQL Database Schema

-- Create database
CREATE DATABASE IF NOT EXISTS traffic_monitoring CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE traffic_monitoring;

-- Table for cameras
CREATE TABLE IF NOT EXISTS cameras (
    camera_id INT PRIMARY KEY AUTO_INCREMENT,
    camera_name VARCHAR(100) NOT NULL,
    camera_type ENUM('lane', 'helmet', 'red_light') NOT NULL,
    location VARCHAR(255),
    status ENUM('active', 'inactive') DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_camera_name_type (camera_name, camera_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Insert default cameras
INSERT INTO cameras (camera_id, camera_name, camera_type, location) VALUES
(1, 'Camera 1', 'lane', 'Ngã tư A - Phát hiện làn đường'),
(2, 'Camera 2', 'helmet', 'Ngã tư B - Phát hiện mũ bảo hiểm'),
(3, 'Camera 3', 'red_light', 'Ngã tư C - Phát hiện vượt đèn đỏ')
ON DUPLICATE KEY UPDATE location=VALUES(location);

-- Table for video uploads
CREATE TABLE IF NOT EXISTS videos (
    video_id INT PRIMARY KEY AUTO_INCREMENT,
    camera_id INT NOT NULL,
    video_filename VARCHAR(255) NOT NULL,
    video_path VARCHAR(500) NOT NULL,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_status ENUM('uploaded', 'processing', 'completed', 'failed') DEFAULT 'uploaded',
    processing_started_at TIMESTAMP NULL,
    processing_completed_at TIMESTAMP NULL,
    file_size_mb DECIMAL(10, 2),
    duration_seconds INT,
    fps INT,
    resolution VARCHAR(20),
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE,
    INDEX idx_camera_upload (camera_id, upload_time),
    INDEX idx_status (processing_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table for lane violations (Camera 1)
CREATE TABLE IF NOT EXISTS lane_violations (
    violation_id INT PRIMARY KEY AUTO_INCREMENT,
    video_id INT NOT NULL,
    camera_id INT NOT NULL DEFAULT 1,
    frame_number INT NOT NULL,
    time_in_video DECIMAL(10, 2) NOT NULL COMMENT 'Time in seconds',
    violation_type ENUM('motor_in_car_lane', 'car_in_motor_lane') NOT NULL,
    vehicle_type VARCHAR(50) NOT NULL,
    confidence DECIMAL(5, 2) NOT NULL,
    bbox_x1 INT,
    bbox_y1 INT,
    bbox_x2 INT,
    bbox_y2 INT,
    image_path VARCHAR(500),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE,
    INDEX idx_video (video_id),
    INDEX idx_camera_time (camera_id, detected_at),
    INDEX idx_violation_type (violation_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table for helmet violations (Camera 2)
CREATE TABLE IF NOT EXISTS helmet_violations (
    violation_id INT PRIMARY KEY AUTO_INCREMENT,
    video_id INT NOT NULL,
    camera_id INT NOT NULL DEFAULT 2,
    frame_number INT NOT NULL,
    time_in_video DECIMAL(10, 2) NOT NULL COMMENT 'Time in seconds',
    has_helmet BOOLEAN NOT NULL DEFAULT FALSE COMMENT 'TRUE = có mũ, FALSE = không mũ (vi phạm)',
    confidence DECIMAL(5, 2) NOT NULL,
    license_plate VARCHAR(50),
    bbox_x1 INT,
    bbox_y1 INT,
    bbox_x2 INT,
    bbox_y2 INT,
    image_path VARCHAR(500),
    pdf_report_path VARCHAR(500),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE,
    INDEX idx_video (video_id),
    INDEX idx_camera_time (camera_id, detected_at),
    INDEX idx_license_plate (license_plate),
    INDEX idx_violation (has_helmet)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table for red light violations (Camera 3)
CREATE TABLE IF NOT EXISTS red_light_violations (
    violation_id INT PRIMARY KEY AUTO_INCREMENT,
    video_id INT NOT NULL,
    camera_id INT NOT NULL DEFAULT 3,
    frame_number INT NOT NULL,
    time_in_video DECIMAL(10, 2) NOT NULL COMMENT 'Time in seconds',
    license_plate VARCHAR(50),
    confidence DECIMAL(5, 2) NOT NULL,
    bbox_x1 INT,
    bbox_y1 INT,
    bbox_x2 INT,
    bbox_y2 INT,
    image_path VARCHAR(500),
    pdf_report_path VARCHAR(500),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE,
    INDEX idx_video (video_id),
    INDEX idx_camera_time (camera_id, detected_at),
    INDEX idx_license_plate (license_plate)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table for processing logs
CREATE TABLE IF NOT EXISTS processing_logs (
    log_id INT PRIMARY KEY AUTO_INCREMENT,
    video_id INT NOT NULL,
    camera_id INT NOT NULL,
    log_level ENUM('INFO', 'WARNING', 'ERROR') DEFAULT 'INFO',
    message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE,
    INDEX idx_video_time (video_id, created_at),
    INDEX idx_level (log_level)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- View for lane violation statistics
CREATE OR REPLACE VIEW v_lane_violation_stats AS
SELECT 
    v.camera_id,
    c.camera_name,
    DATE(lv.detected_at) as violation_date,
    lv.violation_type,
    COUNT(*) as violation_count,
    AVG(lv.confidence) as avg_confidence
FROM lane_violations lv
JOIN videos v ON lv.video_id = v.video_id
JOIN cameras c ON v.camera_id = c.camera_id
GROUP BY v.camera_id, c.camera_name, DATE(lv.detected_at), lv.violation_type;

-- View for helmet violation statistics
CREATE OR REPLACE VIEW v_helmet_violation_stats AS
SELECT 
    v.camera_id,
    c.camera_name,
    DATE(hv.detected_at) as violation_date,
    SUM(CASE WHEN hv.has_helmet = FALSE THEN 1 ELSE 0 END) as no_helmet_count,
    SUM(CASE WHEN hv.has_helmet = TRUE THEN 1 ELSE 0 END) as with_helmet_count,
    COUNT(*) as total_detections,
    AVG(hv.confidence) as avg_confidence
FROM helmet_violations hv
JOIN videos v ON hv.video_id = v.video_id
JOIN cameras c ON v.camera_id = c.camera_id
GROUP BY v.camera_id, c.camera_name, DATE(hv.detected_at);

-- View for red light violation statistics
CREATE OR REPLACE VIEW v_red_light_violation_stats AS
SELECT 
    v.camera_id,
    c.camera_name,
    DATE(rlv.detected_at) as violation_date,
    COUNT(*) as violation_count,
    COUNT(DISTINCT rlv.license_plate) as unique_vehicles,
    AVG(rlv.confidence) as avg_confidence
FROM red_light_violations rlv
JOIN videos v ON rlv.video_id = v.video_id
JOIN cameras c ON v.camera_id = c.camera_id
GROUP BY v.camera_id, c.camera_name, DATE(rlv.detected_at);

-- View for overall statistics
CREATE OR REPLACE VIEW v_overall_stats AS
SELECT 
    c.camera_id,
    c.camera_name,
    c.camera_type,
    COUNT(DISTINCT v.video_id) as total_videos,
    CASE 
        WHEN c.camera_type = 'lane' THEN (SELECT COUNT(*) FROM lane_violations WHERE camera_id = c.camera_id)
        WHEN c.camera_type = 'helmet' THEN (SELECT COUNT(*) FROM helmet_violations WHERE camera_id = c.camera_id AND has_helmet = FALSE)
        WHEN c.camera_type = 'red_light' THEN (SELECT COUNT(*) FROM red_light_violations WHERE camera_id = c.camera_id)
    END as total_violations,
    MAX(v.upload_time) as last_upload_time
FROM cameras c
LEFT JOIN videos v ON c.camera_id = v.camera_id
GROUP BY c.camera_id, c.camera_name, c.camera_type;

-- Sample queries for reporting
-- 1. Get all violations for a specific video
-- SELECT * FROM lane_violations WHERE video_id = ?;
-- SELECT * FROM helmet_violations WHERE video_id = ?;
-- SELECT * FROM red_light_violations WHERE video_id = ?;

-- 2. Get daily violation statistics
-- SELECT * FROM v_lane_violation_stats WHERE violation_date = CURDATE();
-- SELECT * FROM v_helmet_violation_stats WHERE violation_date = CURDATE();
-- SELECT * FROM v_red_light_violation_stats WHERE violation_date = CURDATE();

-- 3. Get overall camera statistics
-- SELECT * FROM v_overall_stats;

SHOW TABLES;
SELECT 'Database setup completed successfully!' as status;
