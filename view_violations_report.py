"""
Công cụ xem và phân tích báo cáo vi phạm làn đường
Tool to view and analyze lane violation reports
"""

import pandas as pd
import os
import json
from datetime import datetime
import glob

def view_latest_violation_report(output_dir="output"):
    """Xem báo cáo vi phạm mới nhất"""
    
    if not os.path.exists(output_dir):
        print(f"❌ Thư mục {output_dir} không tồn tại!")
        return
    
    # Tìm file CSV mới nhất
    csv_files = glob.glob(os.path.join(output_dir, "lane_violations_stats_*.csv"))
    
    if not csv_files:
        print("❌ Không tìm thấy file báo cáo vi phạm nào!")
        return
    
    # Lấy file mới nhất
    latest_csv = max(csv_files, key=os.path.getctime)
    print(f"📊 Đang xem báo cáo: {os.path.basename(latest_csv)}")
    
    try:
        # Đọc dữ liệu CSV
        df = pd.read_csv(latest_csv)
        
        if df.empty:
            print("ℹ️ Không có vi phạm nào trong báo cáo này.")
            return
        
        print(f"\n🚨 TỔNG QUAN VI PHẠM")
        print("=" * 50)
        print(f"Tổng số vi phạm: {len(df)}")
        
        # Thống kê theo loại vi phạm
        violation_counts = df['type'].value_counts()
        print(f"\n📈 THỐNG KÊ THEO LOẠI:")
        for violation_type, count in violation_counts.items():
            type_name = "Xe máy vi phạm làn ô tô" if "xe_may" in violation_type else "Ô tô vi phạm làn xe máy"
            print(f"   {type_name}: {count} lần")
        
        # Top 5 vi phạm theo confidence
        print(f"\n⭐ TOP 5 VI PHẠM ĐỘ TIN CẬY CAO NHẤT:")
        top_violations = df.nlargest(5, 'confidence')[['violation_id', 'type', 'time_formatted', 'confidence', 'vehicle_class']]
        for idx, row in top_violations.iterrows():
            type_name = "Xe máy" if "xe_may" in row['type'] else "Ô tô"
            print(f"   {row['violation_id']:2d}. {type_name} ({row['vehicle_class']}) - {row['time_formatted']} - Độ tin cậy: {row['confidence']:.2f}")
        
        # Thống kê theo thời gian
        print(f"\n⏰ PHÂN BỐ THEO THỜI GIAN:")
        df['minute'] = (df['time_seconds'] // 60).astype(int)
        time_stats = df.groupby('minute').size()
        print(f"Phút có nhiều vi phạm nhất: Phút {time_stats.idxmax()} với {time_stats.max()} vi phạm")
        
        # Thống kê chi tiết
        print(f"\n📋 THÔNG TIN CHI TIẾT:")
        print(f"Độ tin cậy trung bình: {df['confidence'].mean():.3f}")
        print(f"Độ tin cậy cao nhất: {df['confidence'].max():.3f}")
        print(f"Độ tin cậy thấp nhất: {df['confidence'].min():.3f}")
        
        # Kiểm tra file JSON chi tiết
        json_file = latest_csv.replace('_stats_', '_details_').replace('.csv', '.json')
        if os.path.exists(json_file):
            print(f"\n🔍 CHI TIẾT BỔ SUNG TỪ {os.path.basename(json_file)}:")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                video_info = data.get('video_info', {})
                summary = data.get('summary', {})
                
                print(f"   📹 Video gốc: {video_info.get('original_resolution', 'N/A')} @ {video_info.get('fps', 'N/A')}fps")
                print(f"   🎬 Video xử lý: {video_info.get('processed_resolution', 'N/A')}")
                print(f"   ⚡ Performance mode: {video_info.get('performance_mode', 'N/A')}")
                print(f"   ⏱️ Thời gian xử lý: {summary.get('processing_time', 0):.1f}s")
                print(f"   📊 Frame đã xử lý: {video_info.get('processed_frames', 'N/A')}/{video_info.get('total_frames', 'N/A')}")
        
        print(f"\n📁 Vị trí file: {latest_csv}")
        
        return df
        
    except Exception as e:
        print(f"❌ Lỗi khi đọc file báo cáo: {e}")
        return None

def list_all_reports(output_dir="output"):
    """Liệt kê tất cả các báo cáo có sẵn"""
    
    if not os.path.exists(output_dir):
        print(f"❌ Thư mục {output_dir} không tồn tại!")
        return
    
    csv_files = glob.glob(os.path.join(output_dir, "lane_violations_stats_*.csv"))
    video_files = glob.glob(os.path.join(output_dir, "lane_violations_*.mp4"))
    
    if not csv_files and not video_files:
        print("❌ Không tìm thấy báo cáo nào!")
        return
    
    print(f"📂 TẤT CẢ BÁO CÁO TRONG {output_dir}:")
    print("=" * 60)
    
    # Nhóm các file theo timestamp
    reports = {}
    
    for csv_file in csv_files:
        basename = os.path.basename(csv_file)
        timestamp = basename.replace('lane_violations_stats_', '').replace('.csv', '')
        if timestamp not in reports:
            reports[timestamp] = {}
        reports[timestamp]['csv'] = csv_file
    
    for video_file in video_files:
        basename = os.path.basename(video_file)
        timestamp = basename.replace('lane_violations_', '').replace('.mp4', '')
        if timestamp not in reports:
            reports[timestamp] = {}
        reports[timestamp]['video'] = video_file
    
    # Hiển thị theo thứ tự thời gian
    for timestamp in sorted(reports.keys(), reverse=True):
        report = reports[timestamp]
        print(f"\n📅 {timestamp}:")
        
        if 'csv' in report:
            csv_file = report['csv']
            try:
                df = pd.read_csv(csv_file)
                violation_count = len(df)
                print(f"   📊 CSV: {os.path.basename(csv_file)} ({violation_count} vi phạm)")
            except:
                print(f"   📊 CSV: {os.path.basename(csv_file)} (lỗi đọc file)")
        
        if 'video' in report:
            video_file = report['video']
            file_size = os.path.getsize(video_file) / (1024*1024)  # MB
            print(f"   📼 Video: {os.path.basename(video_file)} ({file_size:.1f}MB)")
        
        # Kiểm tra JSON
        json_file = os.path.join(output_dir, f"lane_violations_details_{timestamp}.json")
        if os.path.exists(json_file):
            print(f"   📄 JSON: lane_violations_details_{timestamp}.json")

def export_summary_report(output_dir="output"):
    """Xuất báo cáo tổng hợp từ tất cả các file CSV"""
    
    csv_files = glob.glob(os.path.join(output_dir, "lane_violations_stats_*.csv"))
    
    if not csv_files:
        print("❌ Không tìm thấy file báo cáo nào để tổng hợp!")
        return
    
    all_data = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                # Thêm thông tin source file
                df['source_file'] = os.path.basename(csv_file)
                df['analysis_date'] = os.path.basename(csv_file).replace('lane_violations_stats_', '').replace('.csv', '')
                all_data.append(df)
        except Exception as e:
            print(f"⚠️ Lỗi đọc {csv_file}: {e}")
    
    if not all_data:
        print("❌ Không có dữ liệu hợp lệ để tổng hợp!")
        return
    
    # Gộp tất cả dữ liệu
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Xuất file tổng hợp
    summary_file = os.path.join(output_dir, f"lane_violations_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    combined_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    
    print(f"✅ Đã xuất báo cáo tổng hợp: {summary_file}")
    print(f"📊 Tổng số vi phạm từ {len(csv_files)} file phân tích: {len(combined_df)}")
    
    return summary_file

if __name__ == "__main__":
    print("🚗 CÔNG CỤ XEM BÁO CÁO VI PHẠM LÀN ĐƯỜNG")
    print("=" * 50)
    
    while True:
        print("\nChọn chức năng:")
        print("1. Xem báo cáo mới nhất")
        print("2. Liệt kê tất cả báo cáo")
        print("3. Xuất báo cáo tổng hợp")
        print("0. Thoát")
        
        choice = input("\nNhập lựa chọn (0-3): ").strip()
        
        if choice == "1":
            view_latest_violation_report()
        elif choice == "2":
            list_all_reports()
        elif choice == "3":
            export_summary_report()
        elif choice == "0":
            print("👋 Tạm biệt!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ!")