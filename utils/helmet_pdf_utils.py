import datetime
import os
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from PIL import Image
import cv2


def create_helmet_violation_pdf(info, image_origin, image_violate, output_file_path):
    """
    Tạo biên bản phạt PDF cho vi phạm mũ bảo hiểm
    
    Args:
        info (dict): Thông tin vi phạm
        image_origin (str): Đường dẫn ảnh gốc
        image_violate (str): Đường dẫn ảnh vi phạm
        output_file_path (str): Đường dẫn file PDF đầu ra
    """
    # Tạo đối tượng Canvas với kích thước trang letter
    c = canvas.Canvas(output_file_path, pagesize=letter)

    # Chèn tiêu đề biên bản phạt
    c.setFont("Helvetica-Bold", 18)
    c.drawString(130, 750, "CONG HOA XA HOI CHU NGHIA VIET NAM")
    c.drawString(170, 730, "DOC LAP - TU DO - HANH PHUC")
    c.drawString(210, 700, "BIEN BAN VI PHAM MU BAO HIEM")

    # Chèn thông tin người vi phạm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 660, "Ho va Ten : {}".format(info['name']))
    c.drawString(50, 630, "Dia Chi Thuong Tru : {}".format(info['address']))
    
    # Check if license plate is provided
    license_plate = info.get('license_plate', '...........................................')
    c.drawString(50, 600, "Bien So Xe : {} - THOI GIAN VI PHAM : {}".format(
        license_plate, info['date']))

    # Chèn ảnh vi phạm
    c.drawString(50, 570,
                 "Hinh Anh Vi Pham Tu He Thong Giam Sat Giao Thong TP Ho Chi Minh ( {}".format(info['date']) + " )")
    c.drawImage(image_origin, 50, 300, width=3.5 * inch, height=3.5 * inch)
    c.drawImage(image_violate, 360, 300, width=3 * inch, height=3.5 * inch)

    # Chèn nội dung biên bản phạt
    c.setFont("Helvetica", 12)
    c.drawString(50, 280, "NOI DUNG BIEN BAN PHAT NGUOI :")
    c.drawString(50, 260, "- LOI VI PHAM : {}".format(info['violation']))
    c.drawString(50, 240, "- LOI KHAC (NEU CO) : {}".format(info['violationOther']))
    c.drawString(50, 220, "- MUC PHAT: {} VND".format(info['fine']))
    c.drawString(50, 200, "- HAN NOP PHAT : {}".format(info['deadline']))
    c.drawString(50, 180, "- Y KIEN CUA NGUOI DIEU KHIEN PHUONG TIEN {}".format(info['opinion']))
    c.drawString(100, 130, " NGUOI VI PHAM")
    c.drawString(410, 130, " CAN BO GIAM SAT")

    c.drawString(108, 110, "    KI TEN  ")
    c.drawString(440, 110, "  KI TEN ")

    # Lưu tệp tin PDF
    c.save()


def get_helmet_violation_info(license_plate=None):
    """
    Tạo thông tin mặc định cho biên bản phạt mũ bảo hiểm
    
    Args:
        license_plate (str, optional): Biển số xe nếu có
        
    Returns:
        dict: Thông tin biên bản phạt
    """
    ngay_gio_hien_tai = datetime.datetime.now()
    ngay_gio_dinh_dang = ngay_gio_hien_tai.strftime("%d-%m-%Y %H:%M:%S")
    
    # Thông tin biên bản phạt
    penalty_info = {
        'name': '....................................................................................................................................',
        'address': '....................................................................................................................',
        'date': str(ngay_gio_dinh_dang),
        'violation': 'LOI KHONG DOI MU BAO HIEM TAI DOAN DUONG LY THUONG KIET',
        'violationOther': '....................................................................................................................',
        'fine': ".....................................................................",
        'deadline': '..................................................................',
        'opinion': ': ...................................................................'
    }
    
    # Thêm biển số xe nếu có
    if license_plate:
        penalty_info['license_plate'] = license_plate
    
    return penalty_info


def save_violation_image(frame, violation_count, output_dir="data_xe_vp_bh"):
    """
    Lưu ảnh vi phạm mũ bảo hiểm
    
    Args:
        frame: Frame OpenCV chứa vi phạm
        violation_count (int): Số thứ tự vi phạm
        output_dir (str): Thư mục lưu ảnh
        
    Returns:
        str: Đường dẫn file ảnh đã lưu
    """
    # Tạo thư mục nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # Đường dẫn file ảnh
    image_path = os.path.join(output_dir, f"{violation_count}.jpg")
    
    # Lưu ảnh
    cv2.imwrite(image_path, frame)
    
    return image_path


def create_helmet_pdf_report(frame, violation_count, license_plate=None, output_dir="BienBanNopPhatXeMayViPhamMuBaoHiem"):
    """
    Tạo biên bản phạt PDF hoàn chỉnh cho vi phạm mũ bảo hiểm
    
    Args:
        frame: Frame OpenCV chứa vi phạm
        violation_count (int): Số thứ tự vi phạm
        license_plate (str, optional): Biển số xe nếu có
        output_dir (str): Thư mục lưu PDF
        
    Returns:
        str: Đường dẫn file PDF đã tạo
    """
    # Tạo thư mục nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu ảnh vi phạm
    image_path = save_violation_image(frame, violation_count)
    
    # Tạo file PDF
    pdf_path = os.path.join(output_dir, f"{violation_count}.pdf")
    
    # Tạo thông tin biên bản
    info = get_helmet_violation_info(license_plate)
    
    # Tạo file tạm cho ảnh gốc
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    frame_pil.save(temp_image.name)
    
    try:
        # Tạo PDF
        create_helmet_violation_pdf(info, temp_image.name, image_path, pdf_path)
        print(f"Created helmet violation PDF: {pdf_path}")
        return pdf_path
    finally:
        # Xóa file tạm
        temp_image.close()
        if os.path.exists(temp_image.name):
            os.unlink(temp_image.name)

