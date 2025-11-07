"""
Database Configuration Module
Quản lý kết nối MySQL cho Traffic Monitoring System
"""

import mysql.connector
from mysql.connector import Error
import os
from datetime import datetime
import json

class DatabaseConfig:
    """Database configuration and connection management"""
    
    def __init__(self, host='localhost', user='root', password='', database='traffic_monitoring'):
        """
        Initialize database configuration
        
        Args:
            host: MySQL host (default: localhost)
            user: MySQL user (default: root)
            password: MySQL password
            database: Database name (default: traffic_monitoring)
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                charset='utf8mb4',
                collation='utf8mb4_unicode_ci'
            )
            
            if self.connection.is_connected():
                print(f"✅ Connected to MySQL database: {self.database}")
                return True
        except Error as e:
            print(f"❌ Database connection error: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("🔌 Database connection closed")
    
    def get_connection(self):
        """Get active connection or create new one"""
        if not self.connection or not self.connection.is_connected():
            self.connect()
        return self.connection
    
    def execute_query(self, query, params=None, fetch=False):
        """
        Execute SQL query
        
        Args:
            query: SQL query string
            params: Query parameters (tuple or dict)
            fetch: If True, return results (for SELECT)
        
        Returns:
            For INSERT: last inserted ID
            For SELECT: result rows
            For UPDATE/DELETE: affected rows count
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch:
                results = cursor.fetchall()
                cursor.close()
                return results
            else:
                conn.commit()
                last_id = cursor.lastrowid
                affected = cursor.rowcount
                cursor.close()
                return last_id if last_id > 0 else affected
                
        except Error as e:
            print(f"❌ Query execution error: {e}")
            print(f"Query: {query}")
            if params:
                print(f"Params: {params}")
            return None
    
    def test_connection(self):
        """Test database connection"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            print(f"✅ Database connection test: OK (result={result})")
            return True
        except Error as e:
            print(f"❌ Database connection test failed: {e}")
            return False


class VideoDatabase:
    """Database operations for video management"""
    
    def __init__(self, db_config):
        """
        Initialize with database config
        
        Args:
            db_config: DatabaseConfig instance
        """
        self.db = db_config
    
    def insert_video(self, camera_id, video_filename, video_path, file_size_mb=None, 
                     duration_seconds=None, fps=None, resolution=None):
        """
        Insert new video record
        
        Returns:
            video_id (int) or None if failed
        """
        query = """
            INSERT INTO videos 
            (camera_id, video_filename, video_path, file_size_mb, duration_seconds, fps, resolution)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        params = (camera_id, video_filename, video_path, file_size_mb, duration_seconds, fps, resolution)
        
        video_id = self.db.execute_query(query, params)
        
        if video_id:
            print(f"✅ Video inserted: ID={video_id}, Camera={camera_id}, File={video_filename}")
        else:
            print(f"❌ Failed to insert video: {video_filename}")
        
        return video_id
    
    def update_video_status(self, video_id, status):
        """
        Update video processing status
        
        Args:
            video_id: Video ID
            status: 'uploaded', 'processing', 'completed', 'failed'
        """
        timestamp_field = None
        if status == 'processing':
            timestamp_field = 'processing_started_at'
        elif status in ['completed', 'failed']:
            timestamp_field = 'processing_completed_at'
        
        if timestamp_field:
            query = f"""
                UPDATE videos 
                SET processing_status = %s, {timestamp_field} = NOW()
                WHERE video_id = %s
            """
        else:
            query = """
                UPDATE videos 
                SET processing_status = %s
                WHERE video_id = %s
            """
        
        params = (status, video_id)
        affected = self.db.execute_query(query, params)
        
        if affected:
            print(f"✅ Video {video_id} status updated: {status}")
        
        return affected
    
    def get_video_info(self, video_id):
        """Get video information"""
        query = "SELECT * FROM videos WHERE video_id = %s"
        results = self.db.execute_query(query, (video_id,), fetch=True)
        return results[0] if results else None


class ViolationDatabase:
    """Database operations for violations"""
    
    def __init__(self, db_config):
        self.db = db_config
    
    def insert_lane_violation(self, video_id, frame_number, time_in_video, violation_type,
                              vehicle_type, confidence, bbox=None, image_path=None):
        """Insert lane violation record"""
        query = """
            INSERT INTO lane_violations 
            (video_id, camera_id, frame_number, time_in_video, violation_type, 
             vehicle_type, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, image_path)
            VALUES (%s, 1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox if bbox else (None, None, None, None)
        params = (video_id, frame_number, time_in_video, violation_type, vehicle_type, 
                 confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, image_path)
        
        violation_id = self.db.execute_query(query, params)
        return violation_id
    
    def insert_helmet_violation(self, video_id, frame_number, time_in_video, has_helmet,
                                confidence, license_plate=None, bbox=None, 
                                image_path=None, pdf_report_path=None):
        """Insert helmet violation record"""
        query = """
            INSERT INTO helmet_violations 
            (video_id, camera_id, frame_number, time_in_video, has_helmet, 
             confidence, license_plate, bbox_x1, bbox_y1, bbox_x2, bbox_y2, 
             image_path, pdf_report_path)
            VALUES (%s, 2, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox if bbox else (None, None, None, None)
        params = (video_id, frame_number, time_in_video, has_helmet, confidence, 
                 license_plate, bbox_x1, bbox_y1, bbox_x2, bbox_y2, 
                 image_path, pdf_report_path)
        
        violation_id = self.db.execute_query(query, params)
        return violation_id
    
    def insert_red_light_violation(self, video_id, frame_number, time_in_video, 
                                   license_plate, confidence, bbox=None, 
                                   image_path=None, pdf_report_path=None):
        """Insert red light violation record"""
        query = """
            INSERT INTO red_light_violations 
            (video_id, camera_id, frame_number, time_in_video, license_plate, 
             confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, 
             image_path, pdf_report_path)
            VALUES (%s, 3, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox if bbox else (None, None, None, None)
        params = (video_id, frame_number, time_in_video, license_plate, confidence,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2, image_path, pdf_report_path)
        
        violation_id = self.db.execute_query(query, params)
        return violation_id
    
    def bulk_insert_lane_violations(self, violations_data):
        """
        Bulk insert lane violations for better performance
        
        Args:
            violations_data: List of violation dictionaries
        """
        if not violations_data:
            return 0
        
        query = """
            INSERT INTO lane_violations 
            (video_id, camera_id, frame_number, time_in_video, violation_type, 
             vehicle_type, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, image_path)
            VALUES (%s, 1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        params_list = []
        for v in violations_data:
            bbox = v.get('bbox', [None, None, None, None])
            params = (
                v['video_id'], v['frame_number'], v['time_in_video'],
                v['violation_type'], v['vehicle_type'], v['confidence'],
                bbox[0], bbox[1], bbox[2], bbox[3], v.get('image_path')
            )
            params_list.append(params)
        
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            inserted_count = cursor.rowcount
            cursor.close()
            print(f"✅ Bulk inserted {inserted_count} lane violations")
            return inserted_count
        except Error as e:
            print(f"❌ Bulk insert error: {e}")
            return 0


class StatisticsDatabase:
    """Database operations for statistics"""
    
    def __init__(self, db_config):
        self.db = db_config
    
    def get_lane_stats(self, camera_id=1, date=None):
        """Get lane violation statistics"""
        query = """
            SELECT violation_type, COUNT(*) as count
            FROM lane_violations
            WHERE camera_id = %s
        """
        params = [camera_id]
        
        if date:
            query += " AND DATE(detected_at) = %s"
            params.append(date)
        
        query += " GROUP BY violation_type"
        
        results = self.db.execute_query(query, tuple(params), fetch=True)
        return results
    
    def get_helmet_stats(self, camera_id=2, date=None):
        """Get helmet violation statistics"""
        query = """
            SELECT 
                SUM(CASE WHEN has_helmet = FALSE THEN 1 ELSE 0 END) as no_helmet_count,
                SUM(CASE WHEN has_helmet = TRUE THEN 1 ELSE 0 END) as with_helmet_count,
                COUNT(*) as total_detections
            FROM helmet_violations
            WHERE camera_id = %s
        """
        params = [camera_id]
        
        if date:
            query += " AND DATE(detected_at) = %s"
            params.append(date)
        
        results = self.db.execute_query(query, tuple(params), fetch=True)
        return results[0] if results else None
    
    def get_red_light_stats(self, camera_id=3, date=None):
        """Get red light violation statistics"""
        query = """
            SELECT 
                COUNT(*) as violation_count,
                COUNT(DISTINCT license_plate) as unique_vehicles
            FROM red_light_violations
            WHERE camera_id = %s
        """
        params = [camera_id]
        
        if date:
            query += " AND DATE(detected_at) = %s"
            params.append(date)
        
        results = self.db.execute_query(query, tuple(params), fetch=True)
        return results[0] if results else None
    
    def get_overall_stats(self):
        """Get overall statistics from view"""
        query = "SELECT * FROM v_overall_stats"
        results = self.db.execute_query(query, fetch=True)
        return results


# Helper function to get database instance
def get_database_connection(password=''):
    """
    Get configured database connection
    
    Args:
        password: MySQL root password
    
    Returns:
        DatabaseConfig instance
    """
    db = DatabaseConfig(
        host='localhost',
        user='root',
        password=password,
        database='traffic_monitoring'
    )
    
    if db.connect():
        return db
    else:
        print("⚠️ Failed to connect to database. Please check your MySQL configuration.")
        return None


# Test function
if __name__ == "__main__":
    print("🧪 Testing database connection...")
    
    # Prompt for password
    import getpass
    password = getpass.getpass("Enter MySQL root password: ")
    
    db = get_database_connection(password)
    
    if db:
        # Test connection
        db.test_connection()
        
        # Test video operations
        video_db = VideoDatabase(db)
        test_video_id = video_db.insert_video(
            camera_id=1,
            video_filename='test_video.mp4',
            video_path='/uploads/test_video.mp4',
            file_size_mb=10.5,
            duration_seconds=60,
            fps=30,
            resolution='1920x1080'
        )
        
        if test_video_id:
            print(f"✅ Test video inserted with ID: {test_video_id}")
            
            # Test status update
            video_db.update_video_status(test_video_id, 'processing')
            video_db.update_video_status(test_video_id, 'completed')
        
        # Test statistics
        stats_db = StatisticsDatabase(db)
        overall = stats_db.get_overall_stats()
        print(f"\n📊 Overall Statistics:")
        for row in overall:
            print(f"   {row}")
        
        db.disconnect()
        print("\n✅ All tests completed!")
    else:
        print("\n❌ Database connection failed. Cannot run tests.")
