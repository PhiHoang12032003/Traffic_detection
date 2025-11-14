import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


def bienBanNopPhat(info, image_origin, image_violate, output_file_path):
    """
    Tạo biên bản vi phạm làn đường dưới dạng PDF
    
    Args:
        info: Dictionary chứa thông tin vi phạm
        image_origin: Đường dẫn ảnh gốc (toàn cảnh)
        image_violate: Đường dẫn ảnh vi phạm (vùng cắt)
        output_file_path: Đường dẫn file PDF đầu ra
    """
    # Tạo đối tượng Canvas với kích thước trang letter
    c = canvas.Canvas(output_file_path, pagesize=letter)

    # Chèn tiêu đề biên bản phạt
    c.setFont("Helvetica-Bold", 18)
    c.drawString(130, 750, "CONG HOA XA HOI CHU NGHIA VIET NAM")
    c.drawString(170, 730, "DOC LAP - TU DO - HANH PHUC")
    c.drawString(210, 700, "BIEN BAN VI PHAM")

    # Chèn thông tin người vi phạm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 660, "Ho va Ten : {}".format(info['name']))
    c.drawString(50, 630, "Dia Chi Thuong Tru : {}".format(info['address']))
    
    # Hiển thị biển số xe nếu có (cho tương lai khi tích hợp OCR)
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


def infoObject_motor():
    """
    Tạo template thông tin vi phạm cho xe máy đi vào làn ô tô
    """
    ngay_gio_hien_tai = datetime.datetime.now()
    ngay_gio_dinh_dang = ngay_gio_hien_tai.strftime("%d-%m-%Y %H:%M:%S")
    
    penalty_info = {
        'name': '....................................................................................................................................',
        'address': '....................................................................................................................',
        'date': str(ngay_gio_dinh_dang),
        'violation': 'XE MAY DI VAO LAN OTO TAI DOAN DUONG LY THUONG KIET',
        'violationOther': '....................................................................................................................',
        'fine': ".....................................................................",
        'deadline': '..................................................................',
        'opinion': ': ...................................................................',
        'license_plate': '...........................................'  # Placeholder cho biển số
    }
    return penalty_info


def infoObject_car():
    """
    Tạo template thông tin vi phạm cho ô tô đi vào làn xe máy
    """
    ngay_gio_hien_tai = datetime.datetime.now()
    ngay_gio_dinh_dang = ngay_gio_hien_tai.strftime("%d-%m-%Y %H:%M:%S")
    
    penalty_info = {
        'name': '....................................................................................................................................',
        'address': '....................................................................................................................',
        'date': str(ngay_gio_dinh_dang),
        'violation': 'OTO DI VAO LAN XE MAY TAI DOAN DUONG LY THUONG KIET',
        'violationOther': '....................................................................................................................',
        'fine': ".....................................................................",
        'deadline': '..................................................................',
        'opinion': ': ...................................................................',
        'license_plate': '...........................................'  # Placeholder cho biển số
    }
    return penalty_info


def infoObject():
    """
    Tạo template thông tin vi phạm chung (tương thích với code cũ)
    """
    ngay_gio_hien_tai = datetime.datetime.now()
    ngay_gio_dinh_dang = ngay_gio_hien_tai.strftime("%d-%m-%Y %H:%M:%S")
    
    penalty_info = {
        'name': '....................................................................................................................................',
        'address': '....................................................................................................................',
        'date': str(ngay_gio_dinh_dang),
        'violation': 'DI KHONG DUNG LAN XE QUI DINH TAI DOAN DUONG LY THUONG KIET',
        'violationOther': '....................................................................................................................',
        'fine': ".....................................................................",
        'deadline': '..................................................................',
        'opinion': ': ...................................................................',
        'license_plate': '...........................................'
    }
    return penalty_info
