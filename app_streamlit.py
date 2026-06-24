import streamlit as st
import cv2
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import requests
import os
import time
import math

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Phân Tích Lún Vệt Bánh Xe - Nhóm 10", layout="wide")

# --- KHỞI TẠO BIẾN TOÀN CỤC (SESSION STATE) ---
if 'start_gps' not in st.session_state: st.session_state.start_gps = [21.0055, 105.9334]
if 'end_gps' not in st.session_state: st.session_state.end_gps = [21.0125, 105.9385]
if 'route_coords' not in st.session_state: st.session_state.route_coords = []
if 'route_distance' not in st.session_state: st.session_state.route_distance = 0.0
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = []
if 'map_polylines' not in st.session_state: st.session_state.map_polylines = []
if 'video_path' not in st.session_state: st.session_state.video_path = None

# --- HÀM BỔ TRỢ TOÁN HỌC & BẢN ĐỒ ---
def tinh_khoang_cach(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlam = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def lay_duong_cong_osrm(start, end):
    url = f"https://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=geojson"
    try:
        res = requests.get(url).json()
        if res.get('routes'):
            coords = [[c[1], c[0]] for c in res['routes'][0]['geometry']['coordinates']]
            dist = res['routes'][0]['distance']
            return coords, dist
    except Exception as e:
        pass
    return [], 0.0

def cat_doan_duong_cong(coords, start_m, end_m):
    sub_seg = []
    accumulated = 0.0
    if len(coords) < 2: return sub_seg
    if start_m == 0: sub_seg.append(coords[0])

    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i+1]
        d = tinh_khoang_cach(p1[0], p1[1], p2[0], p2[1])
        next_dist = accumulated + d

        if next_dist >= start_m and accumulated <= end_m:
            if accumulated < start_m and next_dist >= start_m:
                r = (start_m - accumulated) / d
                sub_seg.append([p1[0] + r*(p2[0]-p1[0]), p1[1] + r*(p2[1]-p1[1])])
            if next_dist >= start_m and next_dist <= end_m:
                sub_seg.append(p2)
            if next_dist > end_m and accumulated <= end_m:
                r = (end_m - accumulated) / d
                sub_seg.append([p1[0] + r*(p2[0]-p1[0]), p1[1] + r*(p2[1]-p1[1])])
        accumulated = next_dist
        if accumulated > end_m: break
    return sub_seg

# --- HÀM TÔ MÀU BẢNG SỐ LIỆU THEO MỨC ĐỘ ---
def style_muc_do(val):
    if val == 'Nặng':
        return 'background-color: #fee2e2; color: #b91c1c; font-weight: bold;'
    elif val == 'Trung bình':
        return 'background-color: #ffedd5; color: #c2410c; font-weight: bold;'
    elif val == 'Nhẹ':
        return 'background-color: #fef9c3; color: #a16207; font-weight: bold;'
    elif val == 'Tốt':
        return 'background-color: #d1fae5; color: #047857; font-weight: bold;'
    return ''

# --- HÀM DỰNG BẢN ĐỒ FOLIUM THEO THỜI GIAN THỰC ---
def ve_ban_do_live(key_suffix="static"):
    m = folium.Map(location=st.session_state.start_gps, zoom_start=14)
    folium.Marker(st.session_state.start_gps, popup="Điểm Đầu", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(st.session_state.end_gps, popup="Điểm Cuối", icon=folium.Icon(color='red')).add_to(m)
    
    if st.session_state.route_coords:
        folium.PolyLine(st.session_state.route_coords, color='#64748b', weight=3, opacity=0.5).add_to(m)
    
    for poly in st.session_state.map_polylines:
        folium.PolyLine(poly['coords'], color=poly['color'], weight=6, opacity=0.9, popup=poly['popup']).add_to(m)
        
    return st_folium(m, width="100%", height=400, key=f"map_{key_suffix}")

# --- GIAO DIỆN HEADER ---
st.markdown("<h2 style='text-align: center; color: #b91c1c;'>Hệ Thống Phân Tích Độ Lún Đường Nhựa Qua Video</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #475569;'>ITS - Nhóm 10</p>", unsafe_allow_html=True)
st.divider()

# --- KHU VỰC TẢI FILE VÀ CẤU HÌNH ---
with st.container():
    st.subheader("⚙️ Cấu Hình Tọa Độ & Tải Video")
    col_up1, col_up2, col_up3 = st.columns([1.5, 1, 1])
    
    with col_up1:
        uploaded_file = st.file_uploader("Tải video mặt đường (MP4)", type=["mp4", "avi"])
        vid_id = st.text_input("Mã định danh đoạn đường:", "10_Nhựa_1")
    
    with col_up2:
        st.write("**Chế độ chọn:**")
        pick_mode = st.radio("Thao tác bản đồ:", ["Xem bản đồ", "📍 Cắm Điểm Đầu", "📍 Cắm Điểm Cuối"], horizontal=True)
        start_str = st.text_input("Tọa độ Điểm Đầu (Lat, Lng):", f"{st.session_state.start_gps[0]:.5f}, {st.session_state.start_gps[1]:.5f}")
    
    with col_up3:
        st.write(f"**Chiều dài thực tế:** {st.session_state.route_distance:.1f} mét")
        end_str = st.text_input("Tọa độ Điểm Cuối (Lat, Lng):", f"{st.session_state.end_gps[0]:.5f}, {st.session_state.end_gps[1]:.5f}")
        analyze_btn = st.button("🚀 Bắt Đầu Truyền Luồng & Quét OpenCV", type="primary", use_container_width=True)

# Lắng nghe sự kiện điền tay tọa độ
try:
    s_lat, s_lng = map(float, start_str.split(','))
    if [s_lat, s_lng] != st.session_state.start_gps:
        st.session_state.start_gps = [s_lat, s_lng]
        st.session_state.route_coords, st.session_state.route_distance = lay_duong_cong_osrm(st.session_state.start_gps, st.session_state.end_gps)
        st.rerun()
except: pass

try:
    e_lat, e_lng = map(float, end_str.split(','))
    if [e_lat, e_lng] != st.session_state.end_gps:
        st.session_state.end_gps = [e_lat, e_lng]
        st.session_state.route_coords, st.session_state.route_distance = lay_duong_cong_osrm(st.session_state.start_gps, st.session_state.end_gps)
        st.rerun()
except: pass

if not st.session_state.route_coords and st.session_state.start_gps and st.session_state.end_gps:
    st.session_state.route_coords, st.session_state.route_distance = lay_duong_cong_osrm(st.session_state.start_gps, st.session_state.end_gps)

# --- CHIA ĐÔI BỐ CỤC LÀM VIỆC CHÍNH ---
col_map, col_main = st.columns([4.5, 5.5], gap="medium")

# CỘT TRÁI: BẢN ĐỒ
with col_map:
    st.markdown("<h4 style='color: #1e3a8a;'>🗺️ BẢN ĐỒ GIÁM SÁT TRỰC TUYẾN</h4>", unsafe_allow_html=True)
    map_placeholder = st.empty()
    
    # Nếu chưa bấm nút phân tích, hiển thị bản đồ tĩnh có tính năng cắm điểm tương tác
    if not (analyze_btn and uploaded_file):
        with map_placeholder:
            map_data = ve_ban_do_live("interactive")
            if map_data and map_data.get('last_clicked'):
                clicked = [map_data['last_clicked']['lat'], map_data['last_clicked']['lng']]
                if pick_mode == "📍 Cắm Điểm Đầu":
                    st.session_state.start_gps = clicked
                    st.session_state.route_coords, st.session_state.route_distance = lay_duong_cong_osrm(st.session_state.start_gps, st.session_state.end_gps)
                    st.rerun()
                elif pick_mode == "📍 Cắm Điểm Cuối":
                    st.session_state.end_gps = clicked
                    st.session_state.route_coords, st.session_state.route_distance = lay_duong_cong_osrm(st.session_state.start_gps, st.session_state.end_gps)
                    st.rerun()

# CỘT PHẢI: VIDEO VÀ THỐNG KÊ
with col_main:
    st.markdown("<h4 style='color: #1e3a8a;'>🎬 KẾT QUẢ QUÉT OPENCV</h4>", unsafe_allow_html=True)
    video_placeholder = st.empty()
    st.markdown("<h4 style='color: #1e3a8a; margin-top: 15px;'>📊 BẢNG SỐ LIỆU TỔNG HỢP</h4>", unsafe_allow_html=True)
    table_placeholder = st.empty()
    summary_placeholder = st.empty()

    # Khởi tạo hiển thị bảng trống ban đầu khi chưa phân tích
    if not st.session_state.analysis_results:
        table_placeholder.info("Đang chờ tải video và kích hoạt thuật toán...")

    if analyze_btn and uploaded_file:
        os.makedirs("uploads", exist_ok=True)
        video_path = os.path.join("uploads", uploaded_file.name)
        st.session_state.video_path = video_path
        
        uploaded_file.seek(0)
        with open(video_path, "wb") as f: 
            f.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("❌ Hệ thống không thể đọc được video này!")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            speed_m_s = 15.0  
            estimated_length_meters = (total_frames / fps) * speed_m_s
            
            limit_distance = st.session_state.route_distance if st.session_state.route_distance > 0 else estimated_length_meters
            frames_per_segment = int((50 / speed_m_s) * fps)

            current_segment = 0
            frame_count = 0
            seg_len, seg_wid, seg_area, seg_pos = [], [], [], []
            st.session_state.analysis_results = []
            st.session_state.map_polylines = []

            # Vẽ bản đồ rỗng nền trước khi quét vòng lặp
            with map_placeholder:
                ve_ban_do_live(f"init")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                current_distance_m = int((frame_count / fps) * speed_m_s)
                if current_distance_m > limit_distance: break

                frame_disp = cv2.resize(frame, (640, 360))
                height, width = frame_disp.shape[:2]
                
                mask = np.zeros((height, width), dtype=np.uint8)
                road_polygon = np.array([
                    [int(width * 0.1), int(height * 0.5)],
                    [int(width * 0.9), int(height * 0.5)],
                    [int(width * 1.0), int(height * 1.0)],
                    [int(width * 0.0), int(height * 1.0)]
                ], np.int32)
                cv2.fillPoly(mask, [road_polygon], 255)

                roi_gray = cv2.bitwise_and(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame_disp, cv2.COLOR_BGR2GRAY), mask=mask)

                # --- 1. NHẬN DIỆN NẮP CỐNG HÌNH TRÒN (HOUGH TRÒN) ---
                circles = cv2.HoughCircles(cv2.GaussianBlur(roi_gray, (9, 9), 2), cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=50, minRadius=20, maxRadius=100)
                if circles is not None:
                    for i in np.uint16(np.around(circles))[0, :]:
                        cv2.circle(frame_disp, (i[0], i[1]), i[2], (0, 255, 0), 2)
                        cv2.circle(frame_disp, (i[0], i[1]), 2, (0, 0, 255), 3)

                # --- 2. NHẬN DIỆN VỆT LÚN ---
                contours, _ = cv2.findContours(cv2.Canny(cv2.GaussianBlur(roi_gray, (7, 7), 0), 100, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    if cv2.contourArea(cnt) > 1500:  
                        x, y, w, h = cv2.boundingRect(cnt)
                        L = h * 0.05  
                        W = w * 0.02  
                        A = L * W     
                        
                        center_x = x + (w / 2)
                        if center_x < width / 3: pos = "Trái"
                        elif center_x < 2 * (width / 3): pos = "Giữa"
                        else: pos = "Phải"

                        seg_len.append(L); seg_wid.append(W); seg_area.append(A); seg_pos.append(pos)
                        cv2.rectangle(frame_disp, (x, y), (x + w, y + h), (0, 165, 255), 2)

                cv2.putText(frame_disp, f"QUET: {current_distance_m}m / LIMIT: {int(limit_distance)}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                video_placeholder.image(frame_disp, channels="BGR", use_container_width=True)
                
                frame_count += 1

                # --- 3. KHI CHẠY XONG ĐOẠN 50M -> CẬP NHẬT LIVE CẢ BẢNG LẪN MAP ---
                if frame_count >= frames_per_segment * (current_segment + 1) or frame_count == total_frames:
                    t_len = sum(seg_len)
                    a_wid = sum(seg_wid) / len(seg_wid) if seg_wid else 0
                    t_area = sum(seg_area)
                    
                    unique_positions = list(set(seg_pos))
                    pos_str = ", ".join(unique_positions) if unique_positions else "-"
                    
                    status = "Tốt"
                    map_color = '#10b981' # Xanh lá
                    if t_area > 15.0: 
                        status, map_color = "Nặng", '#ef4444' # Đỏ
                    elif t_area > 5.0: 
                        status, map_color = "Trung bình", '#f97316' # Cam
                    elif t_area > 0: 
                        status, map_color = "Nhẹ", '#eab308' # Vàng

                    seg_name = f"{current_segment*50}-{(current_segment+1)*50}m"
                    
                    # Cập nhật kết quả vào bộ nhớ dòng
                    st.session_state.analysis_results.append({
                        'Phân Đoạn': seg_name,
                        'Dài (m)': round(min(t_len, 50.0), 1),
                        'Rộng (m)': round(a_wid, 2),
                        'Diện tích lún (m²)': round(t_area, 2),
                        'Vị trí': pos_str,
                        'Mức độ': status
                    })

                    # Cắt tọa độ đường cong OSRM tương ứng 50m vừa quét và ném vào Map
                    if st.session_state.route_coords:
                        sub_c = cat_doan_duong_cong(st.session_state.route_coords, current_segment*50, (current_segment+1)*50)
                        if sub_c:
                            st.session_state.map_polylines.append({
                                'coords': sub_c, 'color': map_color, 'popup': f"Đoạn {seg_name}: {status}"
                            })

                    # RENDER LIVE BẢNG ĐÃ TÔ MÀU THEO MỨC ĐỘ
                    df_temp = pd.DataFrame(st.session_state.analysis_results)
                    styled_df = df_temp.style.map(style_muc_do, subset=['Mức độ'])
                    table_placeholder.dataframe(styled_df, use_container_width=True, hide_index=True)

                    # RENDER LIVE LÊN PHÂN VÙNG BẢN ĐỒ (HIỆN MÀU ĐOẠN ĐƯỜNG VỪA CHẠY XONG)
                    with map_placeholder:
                        ve_ban_do_live(f"loop_{current_segment}")
                    
                    current_segment += 1
                    seg_len, seg_wid, seg_area, seg_pos = [], [], [], []

            cap.release()
            st.rerun() 

# --- IN THỐNG KÊ TỔNG QUAN CUỐI CÙNG SAU KHI QUET XONG ---
if st.session_state.analysis_results:
    df = pd.DataFrame(st.session_state.analysis_results)
    styled_df = df.style.map(style_muc_do, subset=['Mức độ'])
    table_placeholder.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    df_rut = df[df['Diện tích lún (m²)'] > 0] 
    tong_so_doan = len(df)
    so_doan_hong = len(df_rut)
    
    ty_le_xuat_hien = (so_doan_hong / tong_so_doan * 100) if tong_so_doan > 0 else 0
    dai_tb = df_rut['Dài (m)'].mean() if so_doan_hong > 0 else 0
    rong_tb = df_rut['Rộng (m)'].mean() if so_doan_hong > 0 else 0
    dt_tb = df_rut['Diện tích lún (m²)'].mean() if so_doan_hong > 0 else 0

    with summary_placeholder.container():
        st.write("---")
        st.markdown(f"**📝 THỐNG KÊ TOÀN TUYẾN ({vid_id}):**")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Tỷ lệ xuất hiện lún", f"{ty_le_xuat_hien:.1f} %")
        sc2.metric("Chiều dài TB", f"{dai_tb:.1f} m")
        sc3.metric("Chiều rộng TB", f"{rong_tb:.2f} m")
        sc4.metric("Diện tích lún TB", f"{dt_tb:.2f} m²")

        # Cấu hình chuỗi ghi vào file báo cáo
        summary_text = f"Mã định danh đoạn đường: {vid_id}\n"
        summary_text += f"Tọa độ Điểm Đầu: {st.session_state.start_gps[0]:.6f}, {st.session_state.start_gps[1]:.6f}\n"
        summary_text += f"Tọa độ Điểm Cuối: {st.session_state.end_gps[0]:.6f}, {st.session_state.end_gps[1]:.6f}\n"
        summary_text += f"THỐNG KÊ TỔNG QUAN: Tỷ lệ lún: {ty_le_xuat_hien:.1f}% | Dài TB: {dai_tb:.1f}m | Rộng TB: {rong_tb:.2f}m | Diện tích lún TB: {dt_tb:.2f}m²\n\n"
        
        csv_data = df.to_csv(index=False)
        final_csv = (summary_text + csv_data).encode('utf-8-sig')

        st.download_button(label="📥 Xuất File Báo Cáo Excel (CSV)", data=final_csv, file_name=f"Bao_Cao_{vid_id}.csv", mime="text/csv")
