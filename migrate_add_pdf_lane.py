"""
Database Migration Script
Add pdf_report_path column to lane_violations table
"""

import mysql.connector
from db_config import get_database_connection

def run_migration():
    """Add pdf_report_path column to lane_violations table"""
    
    print("🔧 Starting database migration...")
    print("📝 Adding pdf_report_path column to lane_violations table")
    
    # Get database password from user
    import getpass
    password = getpass.getpass("Enter MySQL root password: ")
    
    # Get database connection
    db = get_database_connection(password)
    
    if not db:
        print("❌ Failed to connect to database")
        return False
    
    try:
        cursor = db.connection.cursor()
        
        # Check if column already exists
        cursor.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'traffic_monitoring' 
            AND TABLE_NAME = 'lane_violations' 
            AND COLUMN_NAME = 'pdf_report_path'
        """)
        
        exists = cursor.fetchone()[0]
        
        if exists > 0:
            print("✅ Column pdf_report_path already exists in lane_violations table")
            return True
        
        # Add the column
        print("➕ Adding pdf_report_path column...")
        cursor.execute("""
            ALTER TABLE lane_violations 
            ADD COLUMN pdf_report_path VARCHAR(500) NULL AFTER image_path
        """)
        db.connection.commit()
        print("✅ Column added successfully")
        
        # Add index for better performance
        print("📊 Adding index for pdf_report_path...")
        try:
            cursor.execute("""
                CREATE INDEX idx_pdf_report_path ON lane_violations(pdf_report_path)
            """)
            db.connection.commit()
            print("✅ Index created successfully")
        except mysql.connector.Error as idx_err:
            if idx_err.errno == 1061:  # Duplicate key name
                print("ℹ️ Index already exists")
            else:
                print(f"⚠️ Warning: Could not create index: {idx_err}")
        
        # Verify the change
        print("\n📋 Verifying table structure...")
        cursor.execute("DESCRIBE lane_violations")
        columns = cursor.fetchall()
        
        print("\n✅ Current lane_violations table structure:")
        for col in columns:
            print(f"   - {col[0]}: {col[1]} ({col[2]})")
        
        # Check if pdf_report_path is there
        pdf_col = [col for col in columns if col[0] == 'pdf_report_path']
        if pdf_col:
            print(f"\n✅ SUCCESS: pdf_report_path column added!")
            print(f"   Type: {pdf_col[0][1]}")
            print(f"   Null: {pdf_col[0][2]}")
        else:
            print("\n❌ ERROR: Column not found after migration")
            return False
        
        cursor.close()
        return True
        
    except mysql.connector.Error as err:
        print(f"❌ Database error: {err}")
        if db.connection:
            db.connection.rollback()
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if db and db.connection:
            db.connection.close()
            print("\n🔌 Database connection closed")

if __name__ == "__main__":
    print("="*70)
    print("DATABASE MIGRATION: Add pdf_report_path to lane_violations")
    print("="*70)
    
    success = run_migration()
    
    print("\n" + "="*70)
    if success:
        print("🎉 MIGRATION COMPLETED SUCCESSFULLY!")
        print("✅ You can now save PDF report paths for lane violations")
    else:
        print("❌ MIGRATION FAILED!")
        print("⚠️ Please check the error messages above")
    print("="*70)
