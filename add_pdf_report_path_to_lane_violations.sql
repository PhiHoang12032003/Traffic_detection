-- Migration: Add pdf_report_path column to lane_violations table
-- Date: 2025-11-14
-- Purpose: Store PDF report paths for lane violation records

USE traffic_monitoring;

-- Add pdf_report_path column to lane_violations table
ALTER TABLE lane_violations 
ADD COLUMN pdf_report_path VARCHAR(500) NULL AFTER image_path;

-- Add index for better query performance
CREATE INDEX idx_pdf_report_path ON lane_violations(pdf_report_path);

-- Verify the change
DESCRIBE lane_violations;

-- Show sample data
SELECT 
    violation_id,
    video_id,
    frame_number,
    violation_type,
    image_path,
    pdf_report_path
FROM lane_violations
LIMIT 5;
